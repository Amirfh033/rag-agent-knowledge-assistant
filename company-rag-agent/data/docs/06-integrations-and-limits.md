# Integrations and Technical Limits

## Integrations
Nimbus Flow has more than **400 built-in integrations**. Any service with a REST API can be connected through the generic HTTP node. Custom integrations can be written in JavaScript or Python in the Code node.

## API
The Nimbus Flow public API allows creating, starting and monitoring workflows. The rate limit is **100 requests per minute** per workspace on Business and **1,000 requests per minute** on Enterprise. The Starter plan has no API access.

## Webhooks
Each workflow can have a webhook trigger. The maximum webhook payload size is **16 MB**. Webhook URLs can be protected with header authentication or HMAC signatures.

## Execution limits
- Maximum execution time of a single workflow run: **5 minutes** on Starter, **30 minutes** on Business, **2 hours** on Enterprise.
- Maximum memory per execution: 512 MB (Starter), 2 GB (Business), 8 GB (Enterprise).

## AI features
The AI node supports OpenAI, Anthropic, Mistral and self-hosted models through an OpenAI-compatible endpoint. AI token usage is billed by the model provider, not by Nimbus Flow.

## Retries
Every node can be configured to retry on failure up to **5 times** with a configurable wait between attempts.
