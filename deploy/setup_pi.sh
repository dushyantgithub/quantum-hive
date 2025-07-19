#!/bin/bash

# Quantum Hive - Raspberry Pi Setup Script
# This script sets up a Raspberry Pi 4 for running Quantum Hive

set -e  # Exit on any error

echo "🧠 Quantum Hive - Raspberry Pi Setup"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
        print_warning "This script is designed for Raspberry Pi. Continue anyway? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    print_success "System updated"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Audio dependencies
    sudo apt install -y \
        python3-pip \
        python3-venv \
        portaudio19-dev \
        python3-pyaudio \
        espeak-ng \
        espeak-ng-data \
        libespeak-ng-dev \
        libasound2-dev \
        libportaudio2 \
        libportaudiocpp0 \
        ffmpeg \
        sox \
        libsox-fmt-all
    
    # Development tools
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip
    
    # Python development
    sudo apt install -y \
        python3-dev \
        python3-setuptools \
        python3-wheel
    
    print_success "System dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Download Vosk model
download_vosk_model() {
    print_status "Downloading Vosk speech recognition model..."
    
    MODEL_DIR="models"
    MODEL_NAME="vosk-model-small-en-us-0.15"
    MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    
    if [ ! -d "$MODEL_NAME" ]; then
        print_status "Downloading $MODEL_NAME..."
        wget "$MODEL_URL"
        unzip "${MODEL_NAME}.zip"
        rm "${MODEL_NAME}.zip"
        print_success "Vosk model downloaded"
    else
        print_status "Vosk model already exists"
    fi
    
    cd ..
}

# Optimize Raspberry Pi settings
optimize_pi() {
    print_status "Optimizing Raspberry Pi settings..."
    
    # Increase GPU memory
    if ! grep -q "gpu_mem=" /boot/config.txt; then
        echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    fi
    
    # Enable audio
    if ! grep -q "dtparam=audio=on" /boot/config.txt; then
        echo "dtparam=audio=on" | sudo tee -a /boot/config.txt
    fi
    
    # Set CPU governor to performance
    echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
    
    # Disable WiFi power management
    echo "power_save = 0" | sudo tee -a /etc/NetworkManager/conf.d/10-globally-managed-devices.conf
    
    print_success "Raspberry Pi optimized"
}

# Create systemd service
create_service() {
    print_status "Creating systemd service..."
    
    SERVICE_FILE="/etc/systemd/system/quantum-hive.service"
    
    cat > quantum-hive.service << EOF
[Unit]
Description=Quantum Hive AI Assistant
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python backend/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv quantum-hive.service "$SERVICE_FILE"
    sudo systemctl daemon-reload
    sudo systemctl enable quantum-hive.service
    
    print_success "Systemd service created"
}

# Setup audio
setup_audio() {
    print_status "Setting up audio..."
    
    # Add user to audio group
    sudo usermod -a -G audio "$USER"
    
    # Create .asoundrc for better audio configuration
    cat > ~/.asoundrc << EOF
pcm.!default {
    type hw
    card 0
    device 0
}

ctl.!default {
    type hw
    card 0
}
EOF
    
    print_success "Audio configured"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start_quantum_hive.sh << 'EOF'
#!/bin/bash

# Quantum Hive Startup Script
cd "$(dirname "$0")"
source venv/bin/activate
python backend/main.py
EOF
    
    chmod +x start_quantum_hive.sh
    
    print_success "Startup script created"
}

# Main setup function
main() {
    print_status "Starting Quantum Hive setup..."
    
    check_raspberry_pi
    update_system
    install_system_deps
    install_python_deps
    download_vosk_model
    optimize_pi
    setup_audio
    create_service
    create_startup_script
    
    print_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Reboot your Raspberry Pi: sudo reboot"
    echo "2. Test the installation: ./start_quantum_hive.sh"
    echo "3. Enable the service: sudo systemctl start quantum-hive"
    echo "4. Check status: sudo systemctl status quantum-hive"
    echo ""
    echo "For troubleshooting, check the logs:"
    echo "sudo journalctl -u quantum-hive -f"
}

# Run main function
main "$@" 