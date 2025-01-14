# cogstack-rag-search

These are some initial thoughts on how to go about building a RAG based search engine for CogStack at UCLH. 

The purpose of this solution is to provide users (clinicians) with an effective way of searching through Electronic Health Records, particularly for more complex queries that would benefit from using semantic search. This would most likely complement the existing keyword based search through Elasticsearch.

## Solution

### Requirements

We should start by defining functional and non-functional requirements for the system.

Functional requirements include things like specific use cases and query types, expected responses / search results, user experience, factual consistency, harmlessness etc.

Non-functional requirements incudes things like reliability, scalability, latency, throughput, cost, security etc.

### High-Level Design

A basic RAG architecture is shown below. It is split into an offline environment where data is indexed, and an online environment where a user sends a query, followed by document retrieval and response generation.

![RAG pipeline](https://miro.medium.com/v2/resize:fit:828/format:webp/1*FhMJ8OE_PoeOyeAavYjzlw.png)
