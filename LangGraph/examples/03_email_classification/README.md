# Email Classification Agent

This example demonstrates a multi-node workflow for email spam detection and response drafting. Includes implementations for both Claude (Anthropic) and OpenAI.

## Overview

An "Alfred the Butler" themed email processing agent that:
- Reads incoming emails
- Classifies emails as spam or legitimate using LLM
- Routes spam to disposal
- Drafts responses for legitimate emails
- Notifies the user with prepared drafts

## Graph Structure

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌────────────┐
│ read_email │
└─────┬──────┘
      │
      ▼
┌────────────────┐
│ classify_email │
└───────┬────────┘
        │
        ▼ (LLM routing)
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────────┐ ┌────────────────┐
│handle_   │ │ draft_response │
│spam      │ └───────┬────────┘
└────┬─────┘         │
     │               ▼
     │         ┌──────────────┐
     │         │notify_mr_hugg│
     │         └──────┬───────┘
     │                │
     └───────┬────────┘
             ▼
        ┌─────────┐
        │   END   │
        └─────────┘
```

## Files

| File | LLM Provider | Observability |
|------|--------------|---------------|
| `spam_email_agent_claude.py` | Anthropic Claude | Langfuse |
| `spam_email_agent_openai.py` | OpenAI GPT | - |

## Key Features

### 1. State Definition

Comprehensive state tracking for email processing:

```python
class EmailState(TypedDict):
    email: Dict[str, Any]           # Contains subject, sender, body
    email_category: Optional[str]   # inquiry, complaint, thank you, etc.
    spam_reason: Optional[str]      # Why marked as spam
    is_spam: Optional[bool]         # Classification result
    email_draft: Optional[str]      # Generated response
    messages: List[Dict[str, Any]]  # LLM conversation history
```

### 2. LLM-Based Classification

Uses natural language to classify emails:

```python
def classify_email(state: EmailState):
    prompt = f"""
    Analyze this email and determine if it is spam or legitimate.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    """
    response = model.invoke([HumanMessage(content=prompt)])
```

### 3. Conditional Routing

Routes based on spam classification:

```python
def route_email(state: EmailState) -> str:
    if state["is_spam"]:
        return "spam"
    return "legitimate"

email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {"spam": "handle_spam", "legitimate": "draft_response"}
)
```

## Usage

### Process an Email

```python
from spam_email_agent_openai import compiled_graph

email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "I'm interested in learning more about your consulting services."
}

result = compiled_graph.invoke({
    "email": email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

print(f"Is Spam: {result['is_spam']}")
print(f"Category: {result['email_category']}")
print(f"Draft Response: {result['email_draft']}")
```

### Example Outputs

**Spam Email:**
```
Alfred is processing an email from winner@lottery-intl.com...
Alfred has marked the email as spam.
Reason: Unsolicited lottery winnings with request for personal information
The email has been moved to the spam folder.
```

**Legitimate Email:**
```
Sir, you've received an email from john.smith@example.com.
Subject: Question about your services
Category: inquiry

I've prepared a draft response for your review:
--------------------------------------------------
Dear Mr. Smith,

Thank you for reaching out regarding our consulting services...
```

## Environment Variables

### For Claude Version
```bash
ANTHROPIC_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key      # Optional
LANGFUSE_SECRET_KEY=your_key      # Optional
```

### For OpenAI Version
```bash
OPENAI_API_KEY=your_key
```

## Dependencies

### Claude Version
```bash
pip install langgraph langchain-anthropic langfuse
```

### OpenAI Version
```bash
pip install langgraph langchain-openai
```
