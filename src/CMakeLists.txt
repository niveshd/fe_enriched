# src/CMakeLists.txt

# set up shared library by hand:
ADD_LIBRARY(support SHARED support.cc estimate_enrichment.cc)
DEAL_II_SETUP_TARGET(support)

# set up executable with autopilot macro:
SET(TARGET "step")
SET(TARGET_SRC step.cc)
DEAL_II_INVOKE_AUTOPILOT()
TARGET_LINK_LIBRARIES(${TARGET} support)
