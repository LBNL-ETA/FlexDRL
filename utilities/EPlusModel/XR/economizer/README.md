# README

This repository contains scripts to add economizer to the EnergyPlus idf files.

## Structure

``eco_example.idf`` example file to add economizer.

``FlexlabXR_csv_noEco.idf`` Flexlab model without economizer.

``FlexlabXR_csv_Eco.idf`` Flexlab model with economizer.

``test`` Contains simulation results to test the idf files.

## Workflow
Workflow to add an economizer to idf files:

1. add object `Controller:OutdoorAir`
2. add object `Controller:OutdoorAir` to `AirLoopHVAC:ControllerList` class 
3. define new air nodes: `Relief Air Outlet`, `Return Air Inlet`
4. add object `OutdoorAir:Mixer`
5. add the `OutdoorAir:Mixer` object to `AirLoopHVAC:OutdoorAirSystem:EquipmentList`
6. use the `AirLoopHVAC:OutdoorAirSystem:EquipmentList` object and `AirLoopHVAC:ControllerList` object to define `AirLoopHVAC:OutdoorAirSystem`
7. add the `AirLoopHVAC:OutdoorAirSystem` to the Main Air branch in the `branch` class
8. change the `Supply Side Inlet Node Name` in the `AirLoopHVAC` to `Return Air Inlet`
