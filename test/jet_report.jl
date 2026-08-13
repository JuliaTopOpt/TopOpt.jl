"""
    jet_report.jl

Run JET analysis on TopOpt and write a structured report to stdout and a file.
Designed for CI: outputs a diff-friendly report that can be compared across
commits. Non-blocking — intended for PR comments and tracking, not CI gates.

Usage:
    julia --project=test jet_report.jl [--output file.json]
"""

using JET, TopOpt, Dates

output_file = nothing
for arg in ARGS
    if startswith(arg, "--output=")
        global output_file = split(arg, "=")[2]
    elseif startswith(arg, "--output")
        global output_file = ARGS[findfirst(==(arg), ARGS) + 1]
    end
end

# Run JET analysis
report = JET.report_package(TopOpt; target_modules=(TopOpt,))

# Print summary
println("="^60)
println("JET Analysis Report for TopOpt.jl")
println("="^60)
println("Date: ", Dates.now())
println("Julia version: ", VERSION)
println("JET version: ", pkgversion(JET))
println()

# Count issues by type
global n_errors = 0
global n_warnings = 0
all_reports = vcat(report.res.inference_error_reports, report.res.toplevel_error_reports)
for r in all_reports
    if r isa JET.InferenceErrorReport
        global n_errors += 1
    else
        global n_warnings += 1
    end
end

println("Total issues: ", n_errors + n_warnings)
println("  Errors:   ", n_errors)
println("  Warnings: ", n_warnings)
println()

# Print issues grouped by file
if !isempty(all_reports)
    println("Issues by file:")
    by_file = Dict{String,Vector}()
    for r in all_reports
        file = if r isa JET.InferenceErrorReport && !isempty(r.vst)
            string(r.vst[1].file)
        else
            string(r.sig)
        end
        file = replace(file, r"\.jl$" => "")
        file = basename(file)
        push!(get!(by_file, file, []), r)
    end
    for (file, issues) in sort(collect(by_file); by=first)
        println("  $file: $(length(issues)) issues")
    end
    println()

    # Print first 20 issues in detail
    println("Detailed issues (first 20):")
    for (i, r) in enumerate(all_reports)
        i > 20 && break
        println("  [$i] $(r)")
    end
end

# Write JSON report if requested
if output_file !== nothing
    using JSON
    json_report = Dict(
        "date" => string(Dates.now()),
        "julia_version" => string(VERSION),
        "jet_version" => string(pkgversion(JET)),
        "total_issues" => n_errors + n_warnings,
        "errors" => n_errors,
        "warnings" => n_warnings,
        "issues" => [
            Dict(
                "type" => string(typeof(r)),
                "signature" => string(r.sig),
                "message" => string(r),
            ) for r in all_reports
        ],
    )
    write(output_file, JSON.json(json_report, 2))
    println("JSON report written to: ", output_file)
end

# Exit 0 always — JET is non-blocking
println()
println("JET report complete (non-blocking).")
exit(0)
