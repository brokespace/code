import os
import submission
import json

swe_instance = submission.SWE()


def run_swe(repo_location, issue_description):
    return swe_instance(repo_location, issue_description)


if __name__ == "__main__":
    repo_location = "/testbed"
    issue_description = os.getenv("ISSUE_DESCRIPTION")
    result = run_swe(repo_location, issue_description)
    # check if its dumpable, else it's likely a diff
    if hasattr(result, 'model_dump') and callable(result.model_dump):
        print("Patch: ", result.model_dump())
    else:
        print("Diff: ", json.dumps({"diff": result}))
