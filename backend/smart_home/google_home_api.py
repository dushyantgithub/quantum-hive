"""
Google Home API Integration for Quantum Hive
Handles smart home device control through Google Assistant API
"""
import logging
import requests
import json
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class GoogleHomeController:
    """Google Home API controller for smart home devices"""
    
    def __init__(self):
        """Initialize Google Home controller"""
        self.api_key = os.getenv("GOOGLE_HOME_API_KEY")
        self.project_id = os.getenv("GOOGLE_HOME_PROJECT_ID")
        self.device_registry = self._load_device_registry()
        
        if not self.api_key:
            logger.warning("GOOGLE_HOME_API_KEY not found. Using simulation mode.")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            
        logger.info(f"Google Home controller initialized (simulation: {self.simulation_mode})")
    
    def _load_device_registry(self) -> Dict[str, Any]:
        """Load device registry from config file"""
        try:
            config_path = Path(__file__).parent / "device_registry.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading device registry: {e}")
        
        # Default device registry
        return {
            "lights": {
                "living_room": {"id": "living_room_light", "name": "Living Room Light", "type": "light"},
                "bedroom": {"id": "bedroom_light", "name": "Bedroom Light", "type": "light"},
                "kitchen": {"id": "kitchen_light", "name": "Kitchen Light", "type": "light"}
            },
            "thermostat": {
                "main": {"id": "main_thermostat", "name": "Main Thermostat", "type": "thermostat"}
            },
            "fans": {
                "ceiling_fan": {"id": "ceiling_fan_1", "name": "Ceiling Fan", "type": "fan"}
            }
        }
    
    def control_lights(self, action: str, location: str = "all") -> str:
        """Control lights on/off"""
        try:
            if self.simulation_mode:
                return self._simulate_light_control(action, location)
            
            # Real Google Home API call would go here
            # This is a placeholder for the actual implementation
            devices = self._get_light_devices(location)
            command = "action.devices.commands.OnOff"
            params = {"on": action.lower() in ["on", "turn on", "switch on"]}
            
            results = []
            for device in devices:
                result = self._send_device_command(device["id"], command, params)
                results.append(result)
            
            return self._format_light_response(action, location, results)
            
        except Exception as e:
            logger.error(f"Error controlling lights: {e}")
            return f"Sorry Master, I encountered an error controlling the lights: {str(e)}"
    
    def control_temperature(self, action: str, value: Optional[int] = None) -> str:
        """Control thermostat temperature"""
        try:
            if self.simulation_mode:
                return self._simulate_temperature_control(action, value)
            
            # Real Google Home API call would go here
            device = self.device_registry["thermostat"]["main"]
            command = "action.devices.commands.ThermostatTemperatureSetpoint"
            
            if value:
                params = {"thermostatTemperatureSetpoint": value}
            else:
                # Adjust by default amount
                adjustment = 2 if "increase" in action.lower() or "up" in action.lower() else -2
                current_temp = self._get_current_temperature()
                params = {"thermostatTemperatureSetpoint": current_temp + adjustment}
            
            result = self._send_device_command(device["id"], command, params)
            return self._format_temperature_response(action, value, result)
            
        except Exception as e:
            logger.error(f"Error controlling temperature: {e}")
            return f"Sorry Master, I encountered an error controlling the temperature: {str(e)}"
    
    def control_fan(self, action: str, speed: Optional[str] = None) -> str:
        """Control fan on/off and speed"""
        try:
            if self.simulation_mode:
                return self._simulate_fan_control(action, speed)
            
            # Real Google Home API call would go here
            device = self.device_registry["fans"]["ceiling_fan"]
            
            if action.lower() in ["on", "start"]:
                command = "action.devices.commands.OnOff"
                params = {"on": True}
                if speed:
                    command = "action.devices.commands.SetFanSpeed"
                    params = {"fanSpeed": speed}
            else:
                command = "action.devices.commands.OnOff"
                params = {"on": False}
            
            result = self._send_device_command(device["id"], command, params)
            return self._format_fan_response(action, speed, result)
            
        except Exception as e:
            logger.error(f"Error controlling fan: {e}")
            return f"Sorry Master, I encountered an error controlling the fan: {str(e)}"
    
    # Simulation methods for testing without real API
    def _simulate_light_control(self, action: str, location: str) -> str:
        """Simulate light control for testing"""
        logger.info(f"[SIMULATION] Light control: {action} {location}")
        
        if action.lower() in ["on", "turn on", "switch on"]:
            if location.lower() == "all":
                return "All lights have been turned on, Master."
            else:
                return f"The {location} light has been turned on, Master."
        else:
            if location.lower() == "all":
                return "All lights have been turned off, Master."
            else:
                return f"The {location} light has been turned off, Master."
    
    def _simulate_temperature_control(self, action: str, value: Optional[int]) -> str:
        """Simulate temperature control for testing"""
        logger.info(f"[SIMULATION] Temperature control: {action} {value}")
        
        if value:
            return f"Temperature has been set to {value} degrees, Master."
        elif "increase" in action.lower() or "up" in action.lower():
            return "Temperature has been increased by 2 degrees, Master."
        else:
            return "Temperature has been decreased by 2 degrees, Master."
    
    def _simulate_fan_control(self, action: str, speed: Optional[str]) -> str:
        """Simulate fan control for testing"""
        logger.info(f"[SIMULATION] Fan control: {action} {speed}")
        
        if action.lower() in ["on", "start"]:
            if speed:
                return f"Ceiling fan has been turned on at {speed} speed, Master."
            else:
                return "Ceiling fan has been turned on, Master."
        else:
            return "Ceiling fan has been turned off, Master."
    
    # Helper methods for real API integration
    def _get_light_devices(self, location: str) -> List[Dict[str, Any]]:
        """Get light devices based on location"""
        if location.lower() == "all":
            return list(self.device_registry["lights"].values())
        else:
            device = self.device_registry["lights"].get(location.lower())
            return [device] if device else []
    
    def _send_device_command(self, device_id: str, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to Google Home device (placeholder)"""
        # This would be the actual Google Home API call
        # For now, return a mock successful response
        return {
            "device_id": device_id,
            "command": command,
            "params": params,
            "status": "success"
        }
    
    def _get_current_temperature(self) -> int:
        """Get current thermostat temperature (placeholder)"""
        return 72  # Default temperature
    
    def _format_light_response(self, action: str, location: str, results: List[Dict[str, Any]]) -> str:
        """Format light control response"""
        if all(r["status"] == "success" for r in results):
            return self._simulate_light_control(action, location)
        else:
            return "There was an issue controlling some lights, Master."
    
    def _format_temperature_response(self, action: str, value: Optional[int], result: Dict[str, Any]) -> str:
        """Format temperature control response"""
        if result["status"] == "success":
            return self._simulate_temperature_control(action, value)
        else:
            return "There was an issue adjusting the temperature, Master."
    
    def _format_fan_response(self, action: str, speed: Optional[str], result: Dict[str, Any]) -> str:
        """Format fan control response"""
        if result["status"] == "success":
            return self._simulate_fan_control(action, speed)
        else:
            return "There was an issue controlling the fan, Master." 