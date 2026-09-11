# InsightTree

## Adaptive Evidence-Driven Tree Retrieval for Enterprise Analysis

InsightTree is an evidence-driven investigation system designed to answer complex enterprise questions by progressively retrieving, evaluating, and connecting information from heterogeneous data sources.

Instead of relying on a single retrieval step, InsightTree decomposes a complex question into smaller investigation branches, retrieves evidence for each branch, evaluates its quality, identifies information gaps, and iteratively explores the most promising directions.

The system is designed to produce **evidence-grounded, traceable, and citation-supported answers**.

---

## Key Idea

A traditional RAG pipeline generally follows:

    User Question
          ↓
    Retrieve Documents
          ↓
    Generate Answer

This approach can work well for simple questions but becomes less reliable when the answer requires information scattered across multiple documents, numerical analysis, comparison between sources, several reasoning steps, identification of missing information, or investigation of alternative explanations.

InsightTree instead follows an adaptive investigation process:

    User Question
          ↓
    Direct Evidence Retrieval
          │
          ├── Sufficient evidence → Answer
          │
          └── Insufficient evidence
                      ↓
                 Q0 Information Need
                      ↓
              Initial Investigation Questions
                      ↓
                  Branch Exploration
                      ↓
              Evidence Retrieval + Tools
                      ↓
               Evidence Evaluation
                      ↓
              Hypothesis Development
                      ↓
           Information Gap Detection
                      ↓
            New Targeted Questions
                      ↓
              Branch Prioritization
                      ↓
                  Final Synthesis
                      ↓
           Evidence-Grounded Answer

---

## Features

### 1. Direct Query Retrieval

Before starting the full investigation process, InsightTree first searches the indexed knowledge base using the user's original query.

This provides a fast path for questions that can already be answered directly from the available evidence.

If the retrieved evidence is insufficient, the system transforms the question into a structured information need (`Q0`) and starts the adaptive investigation process.

---

### 2. Adaptive Question Tree

Complex questions are decomposed into smaller investigation questions.

Example:

    Q0: Why did the company fail?

    ├── Q1: What happened to the company's revenue?
    ├── Q2: What were the main operational problems?
    ├── Q3: Did the company face financial constraints?
    └── Q4: Were there external market factors?

Each branch can then be explored independently.

The system does not necessarily expand the entire tree uniformly. Instead, it prioritizes branches according to their evidence strength, relevance, and expected information value.

---

### 3. Branch-by-Branch Exploration

InsightTree explores promising branches incrementally.

A branch can generate additional child questions when its current evidence is incomplete.

    Branch Q2
       ↓
    Retrieved evidence
       ↓
    Evidence is incomplete
       ↓
    Generate targeted child questions
       ↓
    Retrieve additional evidence
       ↓
    Update branch conclusion

This avoids generating a large question tree before knowing whether the branches are actually useful.

---

### 4. Hybrid Retrieval

InsightTree combines multiple retrieval strategies:

- Dense vector retrieval
- BM25 lexical retrieval
- Metadata-based filtering
- Evidence ranking

Dense retrieval helps capture semantic similarity, while BM25 helps retrieve exact terms, names, numbers, and domain-specific expressions.

The combination improves retrieval robustness across heterogeneous enterprise documents.

---

### 5. Evidence Evaluation

Retrieved information is not automatically treated as reliable evidence.

Each piece of evidence is evaluated according to several dimensions:

- Relevance
- Source reliability
- Consistency
- Numerical validity
- Completeness
- Citation availability

This allows the system to distinguish between useful evidence and weak or irrelevant retrieval results.

---

### 6. Evidence Action Engine

The system dynamically selects the most appropriate action for each information need.

Available actions include:

    REUSE_EVIDENCE
    SEARCH_DATA
    USE_TOOL
    SEARCH_DATA_AND_USE_TOOL
    MULTI_STEP_ACTION

Example:

    Information Need
           ↓
    Do we already have sufficient evidence?
           │
           ├── Yes → REUSE_EVIDENCE
           ├── Need documents → SEARCH_DATA
           ├── Need calculation → USE_TOOL
           └── Need both → SEARCH_DATA_AND_USE_TOOL

This prevents the system from relying exclusively on retrieval when a deterministic tool would be more appropriate.

---

### 7. Tool Integration

InsightTree can use deterministic tools when required by the investigation.

Examples include:

- Calculator
- Structured data query engine
- External APIs

For example, if the evidence contains:

    Revenue 2023: $12M
    Revenue 2024: $9M

the system can calculate the absolute and relative change instead of asking the language model to estimate it.

---

### 8. Hypothesis Tracking

For complex investigations, InsightTree maintains explicit hypotheses.

Example:

    Hypothesis:
    The company's failure was primarily caused by declining revenue.

    Supporting evidence:
    - Revenue decreased by 25%.
    - Customer acquisition declined.

    Contradicting evidence:
    - Operating costs remained stable.

    Confidence:
    0.78

New evidence can support or contradict existing hypotheses.

This makes the reasoning process more structured and transparent.

---

### 9. Information Gain

The system evaluates whether newly retrieved information provides genuinely new knowledge.

Information that duplicates already-known evidence has lower value than evidence that resolves an important uncertainty.

This helps prioritize investigation branches and avoid unnecessary retrieval.

---

### 10. Branch Prioritization and Pruning

Branches are ranked according to their potential usefulness.

Factors include:

- Evidence strength
- Priority
- Information gain
- Remaining uncertainty
- Relevance to the original question

Low-value branches can be deprioritized or pruned, allowing computational resources to focus on more promising investigation paths.

---

### 11. Cross-Branch Reasoning

After individual branches have been explored, InsightTree combines information across branches.

This is important when the final explanation depends on relationships between several factors.

Example:

    Revenue decline
          +
    Customer loss
          +
    Increasing operating costs
          ↓
    Reduced profitability
          ↓
    Liquidity problems
          ↓
    Business failure

The system can identify these relationships rather than treating each retrieved fact independently.

---

### 12. Evidence-Grounded Synthesis

The final answer is generated using the strongest available evidence.

The synthesis stage is instructed to:

- Avoid unsupported claims
- Preserve exact figures
- Distinguish facts from interpretations
- Report contradictions explicitly
- Identify uncertainty
- Calculate changes when sufficient data is available
- Avoid presenting hypotheses as established facts
- Provide citations to supporting evidence

---

## Main Components

### `InformationNeedService`

Transforms user questions into precise, self-contained information needs.

    User Question
         ↓
    Information Need Q0

It can also transform branch questions and their context into new evidence-oriented information needs.

### `BranchExplorationService`

Responsible for:

- Generating initial investigation questions
- Exploring branches
- Generating targeted child questions
- Filtering redundant or low-value questions

### `EvidenceActionPlannerService`

Determines which action should be performed for a given information need.

    Information Need
           ↓
    Action Planner
           ↓
    Best Available Action

### `EvidenceEvaluatorService`

Evaluates retrieved evidence and estimates:

- Relevance
- Sufficiency
- Information gain
- Evidence quality

### `HypothesisService`

Creates and updates hypotheses using supporting and contradicting evidence.

### `CrossBranchReasoningService`

Analyzes information across multiple branches to identify:

- Global information gaps
- Relationships between findings
- New investigation questions

### `SynthesisService`

Produces the final evidence-grounded answer using:

- Strongest branches
- Validated evidence
- Hypotheses
- Citations

---

## LLM

InsightTree uses a local **Qwen3-8B** model for reasoning and generation.

Running the model locally provides:

- No dependency on paid inference APIs
- Better control over data privacy
- Reproducible experimentation
- Easier experimentation with prompts and reasoning strategies

The LLM is used for tasks such as:

- Information need generation
- Question decomposition
- Evidence evaluation
- Hypothesis generation
- Information gap detection
- Final synthesis

Deterministic operations such as calculations are delegated to dedicated tools whenever possible.

---

## Embeddings

The retrieval layer uses an embedding model to transform documents and queries into vector representations.

The system combines vector similarity with lexical retrieval to support both semantic and exact matching.

    Document
       ↓
    Chunking
       ↓
    Embedding
       ↓
    Vector Index

    User Query
       ↓
    Embedding
       ↓
    Vector Search

The retrieval layer can also combine this with BM25 search:

                    Query
                   /     \
                  ↓       ↓
           Dense Search   BM25
                  \       /
                   ↓     ↓
                  Fusion
                     ↓
              Ranked Evidence

---

## Supported Data Sources

InsightTree is designed for heterogeneous enterprise information sources, including:

- PDF documents
- Word documents
- Excel spreadsheets
- CSV files
- Reports
- Articles
- Structured datasets

Different document types can require different extraction and indexing strategies.

---

## Installation

### 1. Clone the repository

    git clone <repository-url>
    cd InsightTree

### 2. Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

Linux / macOS:

    python3 -m venv .venv
    source .venv/bin/activate

### 3. Install dependencies

    pip install -r requirements.txt

---

## Design Principles

### Evidence Before Generation

The system prioritizes retrieved and validated evidence over unsupported language-model reasoning.

### Adaptive Investigation

The system does not blindly expand the entire question tree. It explores branches according to their usefulness.

### Explicit Uncertainty

Uncertainty and contradictions are preserved rather than hidden.

### Deterministic Tools When Possible

Calculations and structured operations are delegated to deterministic tools instead of relying solely on the LLM.

### Traceability

Final conclusions are linked back to supporting evidence and citations.

### Local-First Architecture

The reasoning model can run locally, which is useful for sensitive enterprise data and experimentation.

---

## Technologies

- Python
- Qwen3-8B
- Ollama
- Dense Vector Retrieval
- Embeddings
- BM25
- Hybrid Search
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Information Gain
- Hypothesis Tracking
- Evidence Evaluation
- Branch Prioritization
- Tool Calling
- Structured Data Querying
- Clean Architecture

---

## Why InsightTree?

Traditional RAG systems are often optimized for retrieving relevant passages and generating an answer.

InsightTree focuses on a different problem:

> **How can an AI system investigate a complex question when the answer is distributed across multiple sources and cannot be obtained from a single retrieval step?**

The project therefore combines retrieval, structured investigation, evidence evaluation, hypothesis management, tool use, and adaptive reasoning into a single workflow.

---

## Future Improvements

Potential improvements include:

- Better multilingual retrieval evaluation
- More advanced reranking
- Improved source reliability estimation
- Automatic document quality assessment
- Persistent investigation memory
- More structured tool orchestration
- Evaluation datasets for multi-hop enterprise questions
- Retrieval and reasoning benchmarks
- Observability and investigation traces
- Parallel branch exploration
- More robust contradiction detection

---

## Project Status

🚧 **In Progress**

InsightTree is currently under **testing and development**. The core architecture and main components are implemented, but the system is not yet finalized.

Current work focuses on testing, evaluation, debugging, and improving the reliability of the retrieval and adaptive investigation workflow.
