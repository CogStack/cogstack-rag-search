Notes from 05/02 meeting
- Start with categories of question - cohort level question, population level, condition level, medication level question etc
- Option to swap in different models - LLM and embedding model
- Vectorizing once

Actions from 05/02 meeting
- Start a doc to capture possible questions - Satyam
- Look for previous ethics submissions - Tom
- Chase Oliver from UCLH digital innovation hub - Satyam
- Look at structured data aspects & Caboodle schema - Satyam

22/01/25 meeting - Suggested Agenda
- Intro and initial thoughts (Satyam, Kawsar)
- Existing KCL research (Shubham, Tom)
- Discussion on scope, existing projects, possible projects & sub-projects, collaboration
- Evaluation

Notes from meeting
- Turn on the rag feature in elastic
- Version on mimic iv
- Extractive vs abstractive questions
- Shubham spoke to clinician at SLAM
- Medical data is tricky to chunk meaningfully - algorithm that does some topic analysis?
- Ideally need some ground truth data - golden dataset with questions (and answers?). This could come from clinicians or an LLM.
- RAGfusion - generating possible queries
- Needs benchmarking
- [EHR-DS-QA](https://physionet.org/content/ehr-ds-qa/1.0.0/)
- Patient level vs. multi-patient queries
- Start with oncology or similar - can target specific clinicians

Actions
- Set up fortnightly catch up - Satyam - DONE
- Use an LLM to generate some questions - Kawsar, Satyam, Shubham?
- Speak to SafEHR team and UCLH digital innovation hub around requirements and types of queries - Satyam, Kawsar
- Think about how to get clinician time / ask of clinicians at KCH, SLaM inc. in specific specialities - Tom?
- Test RAG approach on mimic data - Shubham
- Explore how we can get an omopified version of the mimic data - Kawsar - DONE?
- Provide schema of caboodle to create dummy data - Kawsar
- Build basic front-end MVP using synthetic data to show to stakeholders and use to elicit requirements - Satyam

# cogstack-rag-search

These are some initial thoughts on how to go about building a RAG based search engine for CogStack at UCLH. 

The purpose of this solution is to provide users (clinicians) with an effective way of searching through unstructured data from Electronic Health Records, particularly for more complex queries that would benefit from using semantic / vector search. 

The existing Kibana interface requires training to use, so it is hoped that this solution provides an easier and simpler option.

An example query that we would like to be able to answer - Patients with edometrial cancer with FIGO stage 2, diagnosed in last 6 months. Not recorded in structured data, but likely in unstructured data. There are several components to this including an aggregation aspect, vector matching, date-time filter etc. 

In addition a text-to-SQL component could provide a way to incorporate structured searches over Caboodle as well. This would most likely complement the existing keyword based search through Elasticsearch.

Also want to explore agentic workflows.

## Tasks & Sub-Projects

Research tasks / sub-projects
- Broadly the modelling and evaluation tasks
 - Train own embedding / retrieval model
 - Train own reranking model
 - RLHF / fine-tuning of models using clinician feedback and annotations
 - Algorithm for semantic chunking on medical data

Applied tasks / sub-projects
- Define requirements
- Build application / front-end
- Engineering, deployment and monitoring

## Solution

### Requirements

We should start by defining functional and non-functional requirements for the system.

Functional requirements include things like specific use cases and query types, expected responses / search results, user experience, factual consistency, harmlessness etc.

Non-functional requirements incudes things like reliability, scalability, latency, throughput, cost, security etc.

### High-Level Design

A basic RAG architecture is shown below. It is split into an offline environment where data is indexed, and an online environment where a user sends a query, followed by document retrieval and response generation.

![RAG pipeline](https://miro.medium.com/v2/resize:fit:828/format:webp/1*FhMJ8OE_PoeOyeAavYjzlw.png)

[Source](https://medium.com/@drjulija/what-is-retrieval-augmented-generation-rag-938e4f6e03d1)

Going further than this, some slightly more sophisticated architectures that are often seen in production systems include:

**Two-tower architecture** – adds an additional re-ranking step after doing the retrieval step.

![Two Tower](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F906c3c0f8fe637840f134dbf966839ef89ac7242-3443x1641.png&w=3840&q=75)

[Source](https://www.pinecone.io/learn/series/rag/rerankers/)

**Four-stage architecture** – adds two additional steps of filtering and ordering. The full flow at inference time is i) candidate retrieval, ii) filtering, iii) re-ranking, iv) ordering. 

![Four Stage](https://miro.medium.com/v2/resize:fit:828/format:webp/0*M-kqT5K4y1fSjRY6)

[Source](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e)

We can choose an architecture based on the system requirements and through experimentation.

### Data Sources

- Cogstack as a data source (annotations + free text)
- Caboodle (structured data from hospital)

### Chunking

Chunking involves breaking down documents into small segments (chunks).

There are various chunking strategies that could be employed. Some of these include:
- Fixed size chunking
- Sentence splitting
- Recursive chunking
- Semantic chunking

For more discussion on different chunking strategies see [here](https://www.pinecone.io/learn/chunking-strategies/) and [here](https://research.trychroma.com/evaluating-chunking).

We can experiment with different size chunks and different chunking strategies and see how retrieval performance is affected.

Other things to consider include:
- The nature of the data itself i.e. size / length of typical record
- The type of queries we expect
- The embedding model we use
- The type and length of responses / search results that we want to return
- Cost

### Embeddings

Broadly there are three categories of embedding models we could experiment with:
1.	Use general pre-trained embedding models e.g. [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert), [SentenceTransformers](https://sbert.net/) – these could also serve as a baseline
2.	Use specific medical embedding models e.g. [MedEmbed](https://huggingface.co/abhinand/MedEmbed-base-v0.1)
3.	Train our own embedding model

The choice of embedding model depends on the task, the data, as well as what we want to optimise for e.g. accuracy, latency etc.

Embeddings are often computed using bi-encoder models, which encode queries and documents separately.

The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is a good reference for finding embedding models that perform well at different types of tasks.

### Retrieval

In a two-tower architecture, the candidate retrieval step typically involves retrieving a large-ish number of documents that are most semantically similar to the user’s query.

Typically this is done by storing the embeddings of the documents in a vector DB e.g. [Pinecone](https://www.pinecone.io/) or [pgvector](https://github.com/pgvector/pgvector) and then using a similarity metric such as cosine distance along with an approximate nearest neighbors (ANN) index and search algorithm. Various ANN indices could be used e.g. [FAISS](https://github.com/facebookresearch/faiss), [ScaNN](https://github.com/google-research/google-research/tree/master/scann), or [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/). 

We should also establish a baseline for the retrieval step, which is typically keyword search using an algorithm such as BM25. In this case we can skip the embedding step since BM25 works with sparse vectors.

### Reranking

The reranking step takes the documents that have been retrieved and ranks them, extracting the top-k documents and then passing these to the LLM as context. In general reranking is much slower than retrieval.

One of the benefits of having a separate reranking step is that it allows us to add additional data and features as inputs to the model which can improve performance.

Re-ranking is often done using cross-encoder models, where the query and documents are processed together as input pairs.

There are various other options for re-ranking models. This [post](https://www.galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model) goes into detail on the options and considerations.

One common option is to use a late interaction model such as ColBERT which learns a representation for each token. See [here](https://medium.com/@zz1409/colbert-a-late-interaction-model-for-semantic-search-da00f052d30e) for more details.

Some useful libraries include [RAGatouille](https://github.com/AnswerDotAI/RAGatouille) and [rerankers](https://github.com/AnswerDotAI/rerankers).

### Generation

The final step is to pass the top-k documents to the LLM as context, which the LLM uses to provide a response to the user. 

Various LLMs could be used but since LLama-3.1 is available, that is likely the best option.

Some challenges with using LLMs for generation include the possibility of hallucinations and inconsistencies in search results. These can be managed in various ways, including through prompting, structured outputs and guardrails.

#### Prompting

- Task specific prompts e.g. [GoDaddy](https://www.godaddy.com/resources/news/llm-from-the-trenches-10-lessons-learned-operationalizing-models-at-godaddy#h-1-sometimes-one-prompt-isn-t-enough)
- Prompt templates

#### Structured Outputs

Structured outputs can be used to return responses in a certain format and/or ensure integration with downstream systems.

Tools like [Instructor](https://github.com/instructor-ai/instructor) or [Outlines](https://github.com/dottxt-ai/outlines) can help with this.

#### Guardrails

Guardrails are used to prevent unwanted outputs when generating responses. 

Types of checks may include – output value validation (similar to structured outputs), syntactic checks, semantic checks and safety checks to prevent preventing harmful or dangerous responses.

This can be done via prompting or RLHF, for example see [here](https://arxiv.org/abs/2204.05862).

It can also be done via validation tools such as [Pydantic](https://docs.pydantic.dev/latest/) or the [Guardrails](https://github.com/guardrails-ai/guardrails) package.

Other useful tools include OpenAI’s [content moderation API](https://platform.openai.com/docs/guides/moderation) and packages for detecting personally identifiable information.

### Hybrid Search

The existing Elasticsearch solution still provides many benefits over semantic search, including lower variance, better interpretability and greater computational efficiency. It will also perform better for certain types of queries e.g. searching for names, IDs or acronyms, and allows for metadata to be used to refine results e.g. date filters.

The existing solution can be used as a baseline to compare the RAG-based solution against.

It can also be incorporated as part of a hybrid search system. For example, see [this post](https://www.shortwave.com/blog/deep-dive-into-worlds-smartest-email-ai/#our-most-important-tool-ai-search) from Shortwave. In hybrid systems the keyword search is usually done early in the overall pipeline.

We can use MedCat to extract key entities from the records and store these as metadata alongside the documents, so that documents can also be retrieved through keyword search as part of a pre-filtering step.

Anthropic have recently introduced [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) which is a form of hybrid search & retrieval that incorporates contextual embeddings and contextual BM25.

### Optimisation

There are various ways to optimise RAG pipelines. 

To Be Completed.

#### Agents & Agentic RAG

To Be Completed.

### Evaluation & Testing

There are different methods of evaluating the solution that we could use.

For evaluating the core retrieval tasks, we should start by creating a reference dataset. This would capture query and document (response) pairs. We would then ideally ask clinicians or other experts to annotate these with a score of how relevant the documents are for each query. 

This dataset would then serve as the ground truth for calculating various metrics for each retrieval or re-ranking technique:
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Normalised Discounted Cumulative Gain (nDCG@k)
- Information Density
- Intersection over Union (IoU) for retrieval performance at the token level? See [here](https://research.trychroma.com/evaluating-chunking)

The [ir-measures python package](https://ir-measur.es/en/latest/) could be useful for this.

One thing to be aware of is the trade-off between retrieval recall and LLM recall. See discussion of this [here](https://www.pinecone.io/learn/series/rag/rerankers/). This is one of the reasons to use a reranker.

In addition to this automated evaluation, we may want to do further human evaluation using LGTM@k. This is where we would ask a clinician to manually inspect the top-k results for a query and provide a judgement.

In addition, there are several testing & evaluation mechanisms we should consider:

1.	Business / service metrics – these are dependent on the use cases but should ideally be defined with users and ideally be linked to broader service improvement metrics.
  
2.	Unit tests & assertions e.g. on model outputs when using an LLM for response generation – see [here](https://hamel.dev/blog/posts/evals/#level-1-unit-tests) for examples.

3.	Ongoing user feedback & finetuning – have clinicians provide ongoing feedback directly through the user interface. This can then be used to finetune the models similar to RLHF. See [here](https://arxiv.org/abs/2009.01325) for an example of something like this related to summarisation.

4.	LLM-as-a-Judge – using one LLM to evaluate the responses of another. Shows promise but also need to be mindful of possible biases. See [here](https://arxiv.org/abs/2306.05685) for an evaluation of this technique.

[Ragas](https://docs.ragas.io/en/stable/) for evaluation / LLM-as-a-judge.

## Deployment & Production

### Integration

Integration with ModelGateway / ModelServe?

### Modular Architecture

We should build an architecture that is modular, scalable and maintainable.

The main modules / pipelines could be:
- Ingestion module – to populate the vector DB. This would do the pre-processing, chunking and embedding of documents. This would run on a schedule.
- Retrieval module – this pre-processes the user input and queries the vector DB to retrieve the most similar documents
- Re-ranking module – this ranks the retrieved documents and returns the top-k
- Generation module – this takes the user input and retrieved documents and passes them to the LLM, along with any prompt elements, guardrails etc 

Further guidance on how to build a modular architecture can be found [here](https://decodingml.substack.com/p/rag-fundamentals-first).

Certain modules and pipelines can be run offline, while others depend on the user’s query and therefore need to be run online. Making this distinction can help to reduce latency by precomputing as much as possible.

The offline environment will typically include batch processes such as model training, creating embeddings and building the ANN index.

The online environment will serve individual user requests, which will typically involve pre-processing and embedding the query, candidate retrieval and then reranking.

In addition we may want to consider other modules such as a feature store, model registry and serving layer though this may be provided through ModelGateway / ModelServe?

### Caching

There are various things you might want to cache in a RAG pipeline including the embeddings, document retrieval results and the final generated responses. 

This saves computation time and allows us to serve results and responses immediately when we see the same inputs, leading to reduced latency and cost. See [here](https://medium.com/@praveencs87/caching-in-retrieval-augmented-generation-rag-defdd3a91c9d) for more details.

There are also various options for how to cache e.g. in-memory, database etc.

KV caching and prompt caching?

### Monitoring

Things we may want to monitor and log:
- Prompts and prompt templates
- Input-output pairs
- Traces – see [here](https://hamel.dev/blog/posts/evals/#logging-traces)

### User Experience

As part of the requirements gathering phase, we should define the user experience and design the front-end / user interface. 

Need to set expectations for users through some kind of disclaimer e.g. around possible hallucinations.

Provide attribution e.g. explain why the system did what it did, provide sources & citations where possible.

Provide examples and tips for how to search / prompt the system to get the best results.

## Resources

These are some resources that I've found useful:
- [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
- [Real-time Machine Learning For Recommendations](https://eugeneyan.com/writing/real-time-recommendations/)
- [System Design for Recommendations and Search](https://eugeneyan.com/writing/system-design-for-discovery/)
- [Applied LLMs](https://applied-llms.org/)
- [Retrieval Augmented Generation | Pinecone](https://www.pinecone.io/learn/series/rag/)
- [AIE Book - RAG Resources](https://github.com/chiphuyen/aie-book/blob/main/resources.md?trk=feed_main-feed-card_reshare_feed-article-content#rag)
- [Back to Basics for RAG](https://www.youtube.com/watch?v=nc0BupOkrhI)
- [Retrieval-Augmented Generation (RAG) Fundamentals First](https://decodingml.substack.com/p/rag-fundamentals-first)
- [Beyond the Basics of Retrieval for Augmenting Generation](https://www.youtube.com/watch?v=0nA5QG3087g)
- [Chunking Strategies for LLM Applications | Pinecone](https://www.pinecone.io/learn/chunking-strategies/)
- [Mastering RAG: How to Select A Reranking Model - Galileo AI](https://www.galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model)
- [Search reranking with cross-encoders | OpenAI Cookbook](https://cookbook.openai.com/examples/search_reranking_with_cross-encoders)
- [How To Optimize Your RAG Pipelines - by Damien Benveniste](https://newsletter.theaiedge.io/p/how-to-optimize-your-rag-pipelines)
- [Recommender Systems, Not Just Recommender Models | by Even Oldridge | NVIDIA Merlin | Medium](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e)
- [6 Advanced RAG Optimization Strategies: Analysis of 14 Key Research Papers | by Joyce Birkins | Medium](https://medium.com/@joycebirkins/6-advanced-rag-optimization-strategies-analysis-of-14-key-research-papers-f12329975009)
- [Advanced RAG Optimization Strategies (2): Problem Decomposition, Retrieval Clues, and Step-by-Step Reasoning | by Joyce Birkins | Medium](https://medium.com/@joycebirkins/advanced-rag-optimization-strategies-2-problem-decomposition-retrieval-clues-and-step-by-step-df2ff32fec01)
