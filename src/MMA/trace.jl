macro mmatrace()
    esc(quote
        if tracing
            dt = Dict()
            if model.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(∇f_x)
                dt["λ"] = copy(results.minimizer)
            end
            update!(tr,
                    iter,
                    f_x,
                    gr_residual,
                    dt,
                    model.store_trace,
                    model.show_trace)
        end
    end)
end
