# DRL Thesis


## How to Speedup the simulation:
    1. Unreal - downfacing arrow by the Play button - Advanced Settings - 
       Search for "Use less CPU in the background" and disable it (tick is on)
    2. Unreal - Settings - Engine Scalability Settings - Low
    3. Unreal - Shrink Simulation window inside Unreal Editor as much as possible so the simulation will consume fewer resources
       (thus crashes should be prevented) 
    4. AirSim settings.json - Replace your file with the one in the git repo  (settings - original.json kept for reference)
       If the Unreal Editor Crashes - Decrease ClockSpeed in settings.json - max possible value depends on your HW specs).
       Ido stabilized the simulation at ClockSpeed = 3, but don't be afraid to explore higher speeds.
       If you do not have an Nvidia GPU, delete the blocks starting with GpuId and UseNvidiaHardwareEncoder.
    

## Experiments:

These are the tests

