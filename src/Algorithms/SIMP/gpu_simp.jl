import ..TopOpt: whichdevice

whichdevice(s::SIMPResult) = whichdevice(s.topology)
whichdevice(s::SIMP) = whichdevice(s.optimizer)
whichdevice(c::ContinuationSIMP) = whichdevice(c.simp)
whichdevice(o::MMAOptimizer) = whichdevice(o.model)
whichdevice(o::BoxOptimizer) = whichdevice(o.obj)
