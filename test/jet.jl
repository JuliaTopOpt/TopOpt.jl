using JET, TopOpt

# Run JET analysis on the TopOpt package.
# Start with a permissive config to catch obvious issues without
# blocking CI on known type-instabilities or missing methods.
JET.test_package(TopOpt; target_modules=(TopOpt,))
