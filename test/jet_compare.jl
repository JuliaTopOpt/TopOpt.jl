using JSON

# Read two JET JSON reports and produce a markdown diff.
# Usage: julia jet_compare.jl master.json pr.json [--full]
# With --full, also lists all issues on master and PR. Without it, only the diff.

master_file = ARGS[1]
pr_file = ARGS[2]
full_report = "--full" in ARGS

master = JSON.parsefile(master_file)
pr = JSON.parsefile(pr_file)

function issue_key(issue)
    return issue["type"] * "|" * issue["signature"]
end

master_issues = Dict(issue_key(i) => i for i in master["issues"])
pr_issues = Dict(issue_key(i) => i for i in pr["issues"])

new_issues = [i for k in keys(pr_issues) if !haskey(master_issues, k) for i in [pr_issues[k]]]
fixed_issues = [i for k in keys(master_issues) if !haskey(pr_issues, k) for i in [master_issues[k]]]

# Sort by file for readability
sort!(new_issues; by=i -> i["signature"])
sort!(fixed_issues; by=i -> i["signature"])

master_total = master["total_issues"]
pr_total = pr["total_issues"]
master_errors = master["errors"]
pr_errors = pr["errors"]
master_warnings = master["warnings"]
pr_warnings = pr["warnings"]

lines = String[]
push!(lines, "### JET Report Comparison")
push!(lines, "")
push!(lines, "| | Master | PR | Δ |")
push!(lines, "|---|---|---|---|")
push!(lines, "| Total issues | $(master_total) | $(pr_total) | $(pr_total - master_total) |")
push!(lines, "| Errors | $(master_errors) | $(pr_errors) | $(pr_errors - master_errors) |")
push!(lines, "| Warnings | $(master_warnings) | $(pr_warnings) | $(pr_warnings - master_warnings) |")
push!(lines, "")

if !isempty(new_issues)
    push!(lines, "#### ⚠️ New issues ($(length(new_issues)))")
    push!(lines, "")
    for (i, issue) in enumerate(new_issues)
        sig = issue["signature"]
        msg = issue["message"]
        # Truncate long messages
        if length(msg) > 500
            msg = msg[1:500] * "…"
        end
        push!(lines, "<details><summary>$(i). $(issue["type"]) — $(basename(sig))</summary>")
        push!(lines, "")
        push!(lines, "```")
        push!(lines, msg)
        push!(lines, "```")
        push!(lines, "</details>")
        push!(lines, "")
    end
end

if !isempty(fixed_issues)
    push!(lines, "#### ✅ Fixed issues ($(length(fixed_issues)))")
    push!(lines, "")
    for (i, issue) in enumerate(fixed_issues)
        sig = issue["signature"]
        push!(lines, "<details><summary>$(i). $(issue["type"]) — $(basename(sig))</summary>")
        push!(lines, "")
        push!(lines, "```")
        msg = issue["message"]
        if length(msg) > 500
            msg = msg[1:500] * "…"
        end
        push!(lines, msg)
        push!(lines, "```")
        push!(lines, "</details>")
        push!(lines, "")
    end
end

if isempty(new_issues) && isempty(fixed_issues)
    push!(lines, "No new or fixed issues.")
end

# Full issue lists only when --full is passed (push to master)
if full_report
    all_master_issues = sort(collect(values(master_issues)); by=i -> i["signature"])
    if !isempty(all_master_issues)
        push!(lines, "")
        push!(lines, "#### 📋 All issues on master ($(length(all_master_issues)))")
        push!(lines, "")
        for (i, issue) in enumerate(all_master_issues)
            sig = issue["signature"]
            msg = issue["message"]
            if length(msg) > 500
                msg = msg[1:500] * "…"
            end
            push!(lines, "<details><summary>$(i). $(issue["type"]) — $(basename(sig))</summary>")
            push!(lines, "")
            push!(lines, "```")
            push!(lines, msg)
            push!(lines, "```")
            push!(lines, "</details>")
            push!(lines, "")
        end
    end

    all_pr_issues = sort(collect(values(pr_issues)); by=i -> i["signature"])
    if !isempty(all_pr_issues)
        push!(lines, "")
        push!(lines, "#### 📋 All issues in PR ($(length(all_pr_issues)))")
        push!(lines, "")
        for (i, issue) in enumerate(all_pr_issues)
            sig = issue["signature"]
            msg = issue["message"]
            if length(msg) > 500
                msg = msg[1:500] * "…"
            end
            push!(lines, "<details><summary>$(i). $(issue["type"]) — $(basename(sig))</summary>")
            push!(lines, "")
            push!(lines, "```")
            push!(lines, msg)
            push!(lines, "```")
            push!(lines, "</details>")
            push!(lines, "")
        end
    end
end

write("jet_diff.md", join(lines, "\n"))
println(join(lines, "\n"))
