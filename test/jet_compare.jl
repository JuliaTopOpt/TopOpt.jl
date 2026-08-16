using JSON

# Produce the JET markdown comment.
# Usage: julia jet_compare.jl master.json pr.json [--full]
#   Without --full (PR): a Master/PR comparison table plus new/fixed issues.
#   With --full (push to master): just the current issues, or a "no issues" note.

master_file = ARGS[1]
pr_file = ARGS[2]
list_mode = "--full" in ARGS

master = JSON.parsefile(master_file)
pr = JSON.parsefile(pr_file)

function issue_key(issue)
    return issue["type"] * "|" * issue["signature"]
end

master_issues = Dict(issue_key(i) => i for i in master["issues"])
pr_issues = Dict(issue_key(i) => i for i in pr["issues"])

function issue_block(i, issue)
    sig = issue["signature"]
    msg = issue["message"]
    if length(msg) > 500
        msg = msg[1:500] * "…"
    end
    return [
        "<details><summary>$(i). $(issue["type"]) — $(basename(sig))</summary>",
        "",
        "```",
        msg,
        "```",
        "</details>",
        "",
    ]
end

lines = String[]

if list_mode
    issues = sort(collect(values(pr_issues)); by=i -> i["signature"])
    push!(lines, "### JET Report")
    push!(lines, "")
    if isempty(issues)
        push!(lines, "No JET issues found.")
    else
        push!(lines, "$(length(issues)) issue(s):")
        push!(lines, "")
        for (i, issue) in enumerate(issues)
            append!(lines, issue_block(i, issue))
        end
    end
else
    new_issues = [
        i for k in keys(pr_issues) if !haskey(master_issues, k) for i in [pr_issues[k]]
    ]
    fixed_issues = [
        i for k in keys(master_issues) if !haskey(pr_issues, k) for i in [master_issues[k]]
    ]
    sort!(new_issues; by=i -> i["signature"])
    sort!(fixed_issues; by=i -> i["signature"])

    master_total = master["total_issues"]
    pr_total = pr["total_issues"]
    master_errors = master["errors"]
    pr_errors = pr["errors"]
    master_warnings = master["warnings"]
    pr_warnings = pr["warnings"]

    push!(lines, "### JET Report Comparison")
    push!(lines, "")
    push!(lines, "| | Master | PR | Δ |")
    push!(lines, "|---|---|---|---|")
    push!(
        lines,
        "| Total issues | $(master_total) | $(pr_total) | $(pr_total - master_total) |",
    )
    push!(
        lines, "| Errors | $(master_errors) | $(pr_errors) | $(pr_errors - master_errors) |"
    )
    push!(
        lines,
        "| Warnings | $(master_warnings) | $(pr_warnings) | $(pr_warnings - master_warnings) |",
    )
    push!(lines, "")

    if !isempty(new_issues)
        push!(lines, "#### ⚠️ New issues ($(length(new_issues)))")
        push!(lines, "")
        for (i, issue) in enumerate(new_issues)
            append!(lines, issue_block(i, issue))
        end
    end

    if !isempty(fixed_issues)
        push!(lines, "#### ✅ Fixed issues ($(length(fixed_issues)))")
        push!(lines, "")
        for (i, issue) in enumerate(fixed_issues)
            append!(lines, issue_block(i, issue))
        end
    end

    if isempty(new_issues) && isempty(fixed_issues)
        push!(lines, "No new or fixed issues.")
    end
end

write("jet_diff.md", join(lines, "\n"))
println(join(lines, "\n"))
