# src/CMakeLists.txt

# set up executable with autopilot macro:
SET(TARGET "step")
SET(TARGET_SRC step.cc)
DEAL_II_INVOKE_AUTOPILOT()
