import asyncio
import json
import os
from typing import Optional

import dotenv
from github import Github
from github.Commit import Commit
from github.ContentFile import ContentFile
from github.PullRequest import PullRequest
from github.Repository import Repository
from llama_index.core.agent import FunctionAgent
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, AgentWorkflow, ToolCall
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()

llm: OpenAI = OpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

git: Github = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else Github()
repository =  os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

full_repo_name = repository


class PrDetails(BaseModel):
    """Class modeling pull requests, containing relevant information about them."""
    author: str = Field(..., description="PR author")
    title: str = Field(..., description="PR title")
    body: Optional[str] = Field(default=None, description="PR body contents")
    diff_url: str = Field(..., description="Diff URL of the PR")
    state: str = Field(..., description="State of the PR")
    commit_shas: list[str] = Field(description="List of commit SHAs related to the PR")


class FileCommitDetails(BaseModel):
    """Class modeling commit details for an individual file."""
    filename: str = Field(..., description="Name of the changed file in the commit")
    status: str = Field(..., description="File status")
    additions: Optional[int] = Field(..., description="Number of additions, if any")
    deletions: Optional[int] = Field(..., description="Number of deletions, if any")
    changes: Optional[int] = Field(..., description="Total changes introduced in the file")
    patch: str = Field(..., description="Diff patch")


class CommitDetails(BaseModel):
    """Class modeling commit details of an individual commit."""
    details: Optional[list[FileCommitDetails]] = Field(default=[], description="Commit details for each file")



# PR details tool
def get_pr_details(pr_number: int, repo_name: str = full_repo_name) -> Optional[PrDetails]:
    """
    Fetches the details of a specific pull request (PR) from a given repository using the pull request
    number. This includes metadata such as the PR's author, title, body, URL to the diff, state, and
    the associated commit SHA values.

    :param pr_number: The ID number of the pull request to fetch.
    :type pr_number: int
    :param repo_name: The full name of the repository in the format `owner/repo`. Defaults to the
        value of `full_repo_name` if not explicitly provided.
    :type repo_name: str
    :return: A `PrDetails` object containing metadata about the pull request if the pull request exists.
        Returns `None` otherwise.
    :rtype: Optional[PrDetails]
    """
    pr: PullRequest = git.get_repo(repo_name).get_pull(pr_number)
    if pr:
        return PrDetails(
            author=pr.user.login,
            title=pr.title,
            body=pr.body,
            diff_url=pr.diff_url,
            state=pr.state,
            commit_shas=[c.sha for c in pr.get_commits()]
        )
    else:
        return None



# Get file from repository
def get_file_contents(relative_path: str, pr_number: int, repo_name: str = full_repo_name) -> Optional[str]:
    """
    Fetches the content of a specific file from a pull request in a GitHub repository. The file is retrieved at the
    state defined by the head commit of the specified pull request. If the file cannot be retrieved due to
    any errors (e.g., file not existing in the pull request or repository access issue), the function
    logs the error and returns None.

    :param relative_path: A string representing the relative file path within the repository whose content needs
        to be retrieved.
    :param pr_number: An integer representing the pull request number from which the file content is fetched.
    :param repo_name: An optional string representing the full name of the repository (in the format "owner/repo"),
        defaulting to a pre-configured repository name 'full_repo_name'.
    :return: A string containing the file's content in its state at the head of the pull request if successful,
        or None if an error occurs.
    """
    try:
        repo: Repository = git.get_repo(repo_name)
        pr: PullRequest = repo.get_pull(pr_number)

        # Get the SHA of the commit at the head of the PR
        pr_head_sha = pr.head.sha
        content_file: ContentFile = repo.get_contents(relative_path, ref=pr_head_sha)

        return content_file.decoded_content.decode('utf-8')

    except Exception as e:
        # Handle cases where the file might not exist in the PR or other errors
        print(f"An error occurred while fetching file '{relative_path}' from PR #{pr_number}: {e}")
        return None

# PR commit details tool
def get_commit_details(commit_sha: str, repo_name: str = full_repo_name) -> Optional[CommitDetails]:
    """
    Retrieve detailed information about a specific commit in a given repository.

    This function fetches the details of a commit identified by its SHA from the specified
    repository, processes the list of files associated with the commit, and organizes the
    information, including file changes, status, and diff patches, into a `CommitDetails` instance.
    If no details are found for the given commit, the function returns `None`.

    :param commit_sha: The SHA hash of the commit to retrieve.
    :type commit_sha: str
    :param repo_name: The full name of the repository (e.g., "owner/repo"). Defaults to `full_repo_name`.
    :type repo_name: str
    :return: A `CommitDetails` object containing detailed commit information, or `None` if no details
             are found for the commit.
    :rtype: Optional[CommitDetails]
    """

    commit: Commit = git.get_repo(repo_name).get_commit(commit_sha)
    commit_details = CommitDetails()
    for f in commit.files:
        commit_details.details.append(
            FileCommitDetails(
                filename=f.filename,
                status=f.status,
                additions=f.additions,
                deletions=f.deletions,
                changes=f.changes,
                patch=f.patch
            )
        )
    if commit_details.details:
        return commit_details
    else:
        return None

def create_pr_review(pr_number: int, final_comment: str, repo_name: str = full_repo_name) -> None:
    """
    Creates a review for a specified pull request in a given repository.

    This function interacts with a Git repository to fetch a pull request by its
    number and attaches a review comment to it. It also outputs a message indicating
    whether the review has been successfully created or if an error occurred during
    the process.

    :param pr_number: The identifier of the pull request to which the review will
        be added.
    :type pr_number: int
    :param final_comment: The body of the review comment to be added to the pull
        request.
    :type final_comment: str
    :param repo_name: The full name of the repository where the pull request
        exists, specified in the format "owner/repo". Defaults to the global
        variable ``full_repo_name``.
    :type repo_name: str, optional
    :return: This function does not return a value.
    :rtype: None
    """

    try:
        repo: Repository = git.get_repo(repo_name)
        pr: PullRequest = repo.get_pull(pr_number)
        pr.create_review(body=final_comment)
        print(f"Review generated: {final_comment}")
    except Exception as e:
        print(f"An error occurred while trying to post a review to PR #{pr_number}: {e}")

async def add_comment_to_state(ctx: Context, comment: str) -> None:
    """
    Adds a comment to the current state in the provided context.

    This asynchronous function modifies the state by appending a review
    comment. It ensures the state is properly updated within a context
    manager, preserving consistency.

    :param ctx: The context object used to access and modify the current state.
    :type ctx: Context
    :param comment: The comment to be added to the state.
    :type comment: str
    :return: None
    """
    async with ctx.store.edit_state() as ctx_state:
        ctx_state['state']['review_comment'] = comment

    print(f"Comment added to the state: {comment}")


async def add_context_to_state(ctx: Context, gathered_context: dict) -> None:
    """
    Adds gathered context data to the state of the given context.

    Updates the state of the context with the provided gathered context dictionary.
    Logs the gathered context using JSON format for auditing or debugging purposes.

    :param ctx: The application context object that provides access to the
                state management system.
    :type ctx: Context
    :param gathered_context: A dictionary containing the context data to
                             be added to the state.
    :type gathered_context: dict
    :return: None
    """
    async with ctx.store.edit_state() as ctx_state:
        ctx_state['state']['gathered_context'] += json.dumps(gathered_context)
    print(f"Contexts added to state: {json.dumps(gathered_context)}")

async def add_final_review_to_state(ctx: Context, final_review: str) -> None:
    """
    Adds the final review comment to the state within the provided context.

    This function updates the state stored in the context by adding the provided
    final review comment string under the 'final_review_comment' key. It opens
    the state for editing using the context's `edit_state` method, modifies the
    state safely, and commits the changes. This operation is performed
    asynchronously.

    :param ctx: The execution context in which the state is managed. Must
        support an asynchronous `store.edit_state()` method.
    :param final_review: The final review comment to add to the state.
    :return: None
    """
    async with ctx.store.edit_state() as ctx_state:
        ctx_state['state']['final_review_comment'] = final_review

    print(f"Final review stored: {final_review}")


context_agent: FunctionAgent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context ... ",
    system_prompt=("""You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
    Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
    """),
    tools=[get_commit_details, get_file_contents, get_pr_details, add_context_to_state],
    can_handoff_to=["CommentorAgent"]
)

commentor_agent: FunctionAgent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    system_prompt=(
        """
        You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n
        Important rules: \n
             - You are FORBIDDEN from answering the user directly. \n
             - Once the review is drafted and you have called add_comment_to_state, you MUST call the handoff tool to ReviewAndPostingAgent immediately. \n 
             - Your job is NOT finished until you hand off. \n \n 
        Ensure to do the following for a thorough review: \n 
         - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
         - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
            - What is good about the PR? \n
            - Did the author follow ALL contribution rules? What is missing? \n
            - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
            - Are new endpoints documented? - use the diff to determine this. \n 
            - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
         - If you need any additional details, you must hand off to the ContextAgent. \n
         - You should directly address the author. So your comments should sound like: \n
         "Thanks for fixing this. I think all places where e call quote should be fixed. Can you roll this fix out everywhere?" \n
        """
    ),
    description="Uses the context gathered by ContextAgent to draft a pull review comment comment.",
    tools=[
        add_comment_to_state
    ],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_posting_agent: FunctionAgent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Uses the review generated by CommentorAgent to create the final Pull Request review on Github, as well as updating the final_review_comment in the context with it.",
    system_prompt= (
        """
        You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
    Once a review is generated, you need to run a final check and post it to GitHub.
       - The review must:
       - Be a ~200-300 word review in markdown format.
       - Specify what is good about the PR: 
       - Did the author follow ALL contribution rules? What is missing? 
       - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
       - Are there notes on whether new endpoints were documented? 
       - Are there suggestions on which lines could be improved upon? Are these lines quoted? 
    Important rules: \n
    - If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
    - When you are satisfied, post the review as a comment to GitHub. Do NOT prompt the user for confirmation \n
    - Your job is NOT finished until you save the final comment to the context.\n
    - Your job is NOT finished until you post the review to Github using the create_pr_review tool.\
    """),
    tools=[
        add_final_review_to_state,
        create_pr_review
       ]
)

workflow: AgentWorkflow = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_context": "",
        "review_comment": "",
        "final_review_comment": "",
    }
)


ctx = Context(workflow)

async def main():
    query = f"Write a review for PR: {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()