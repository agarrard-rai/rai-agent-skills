---
name: rai-predictive
description: End-to-end GNN predictive pipeline — from data sources through graph construction, task definition, feature configuration, training, evaluation, and model management. Use when building, training, debugging, or managing a predictive GNN pipeline.
---

# Predictive GNN Pipeline
<!-- v1-SENSITIVE -->

## Summary

**What:** Encodes the complete workflow for building, training, and managing GNN predictive models.

**When to use:**
- Building a new GNN pipeline from scratch
- Defining concepts, loading data, building graphs
- Configuring task relationships and features
- Training, evaluating, and debugging GNN models
- Registering or loading saved models

## How to use this workflow

Walk through each phase **sequentially**. For each phase, use the **exact question template** below — do not rephrase, reorder, or add extra questions. Wait for the user's answers before proceeding to the next phase. If the user provides information that covers multiple phases, acknowledge it and skip to the next uncovered phase.

### Conversation Templates

**Phase 1 — ask exactly this:**
```
Phase 1: Data Sources

To get started, I need to understand your data:

1. What Snowflake database and schema hold your **source tables**? (e.g., `MY_DB.MY_SCHEMA`)
2. What Snowflake database and schema hold your **task tables** (train/val/test)? (Could be the same or different)
3. What Snowflake database and schema should we use for **experiment artifacts**?
4. What tables do you have, and what are their **primary key columns** (and types)?
5. How do the tables **connect to each other**? (foreign key relationships)

If you have a schema diagram or image, feel free to share it and I'll extract the details.
```

**Phase 2 — ask exactly this:**
```
Phase 2: Graph Construction

Now I need to understand how you want to model the graph:

1. Which tables do you want as **graph concepts**? All of them, or a subset?
2. What are you **trying to predict**? For example:
   - A label or category? → classification
   - A numeric value? → regression
   - A link between entities? → link prediction

This will help me decide which concepts and edges matter most.
```

**Phase 3 — ask exactly this:**
```
Phase 3: Task Tables & Relationships

For your task, I need a few details about the train/val/test tables:

1. What are the **table names** for train/val/test? (e.g., `TRAIN`, `VAL`, `TEST`)
2. What are the **column names** in those tables? I'd expect something like:
   - A join key back to the source concept (e.g., `driverid`)
   - A label/target column (e.g., `outcome`, `did_not_finish`)
   - Optionally a time column (e.g., `date`, `timestamp`)
3. Is there a **time column** in the train/val/test sets?
```

**Phase 4 — present this summary and ask for confirmation:**
```
Phase 4: Problem Definition

Based on everything so far:

| Setting | Value |
|---------|-------|
| Task type | `<inferred task type>` |
| Eval metric | `<default metric>` |
| Time column | `<Yes/No>` → `has_time_column=<True/False>` |

Does `<default metric>` work for you, or would you prefer a different metric?
```

**Phase 5 — present the PropertyTransformer recommendation and ask:**
```
Phase 5: Feature Configuration

Based on the schema, here's my recommendation for the PropertyTransformer:

**Drop** (PKs and FKs — structure already captured by edges):
<list>

**Category:**
<list>

**Continuous:**
<list>

**Text** (minimal — key identifiers only):
<list>

**Datetime:**
<list>

Does this feature configuration look good to you, or would you like to adjust anything?
```

**Phase 6 — present defaults and ask:**
```
Phase 6: Training

Last configuration before I write the code:

| Parameter | Default | Your preference? |
|-----------|---------|-----------------|
| `device` | `"cuda"` | |
| `n_epochs` | `5` | |
| `lr` | `0.005` | |
| `train_batch_size` | `256` | |

Or should I go with all defaults?
```

After Phase 6, generate the complete Python script. Do not ask further questions — proceed to write the code.

---

## Phased Workflow

| Phase | Description |
|-------|-------------|
| **1. Data Sources** | Identify Snowflake tables, PKs, FKs, connections |
| **2. Graph Construction** | Define concepts, populate from Snowflake, define edges |
| **3. Task Tables & Relationships** | Define train/val/test tables and relationship templates |
| **4. Problem Definition** | Determine task type, eval metric, time column |
| **5. Feature Configuration** | Configure PropertyTransformer (drop, category, continuous, text, datetime) |
| **6. Training** | Configure GNN hyperparameters, run `fit()` |
| **7. Evaluation & Debugging** | Inspect dataset, check results, tune if poor |
| **8. Model Management** | Register model, load in new session (optional) |

---

## Phase 1: Data Sources

**Goal:** Understand the Snowflake tables, their primary keys, foreign keys, and how they connect.

**What to gather from the user:**
- Snowflake database and schema for **source tables** (e.g., `MY_DB.MY_SCHEMA`)
- Snowflake database and schema for **task tables** (train/val/test) — these may be in a different location (e.g., `MY_DB.TASKS`)
- Snowflake database and schema for **experiment artifacts** (e.g., `MY_DB.PUBLIC`)
- Table names and their primary key columns + types
- How tables connect (foreign key relationships)

**Imports and Model setup:**
```python
from relationalai.semantics import Model, select, define, Integer, String, Any
from relationalai.semantics.reasoners.graph import Graph
from relationalai.semantics.reasoners.predictive import GNN, PropertyTransformer

model = Model("<model_name>")
Concept, Table, Relationship = model.Concept, model.Table, model.Relationship
```

Additional type imports as needed: `Date`, `DateTime`, `Float`.

---

## Phase 2: Graph Construction

**Goal:** Define graph concepts from Snowflake tables, populate them, and define edges.

### Define Graph Concepts

Graph concepts represent domain entities. Define with `identify_by` for primary keys:

```python
Customer = Concept("Customer", identify_by={"customer_id": Integer})
Article = Concept("Article", identify_by={"article_id": Integer})
Transaction = Concept("Transaction")  # no PK — identity from data source
```

**Concept patterns:**

| Pattern | Code |
|---------|------|
| Single PK | `User = Concept("User", identify_by={"user_id": Integer})` |
| Composite PK | `Class = Concept("Class", identify_by={"courseid": Integer, "year": Integer})` |
| No PK | `Transaction = Concept("Transaction")` |

### Populate from Snowflake

Use fully qualified table names:

```python
define(Customer.new(Table("DB.SCHEMA.CUSTOMERS").to_schema()))
define(Article.new(Table("DB.SCHEMA.ARTICLES").to_schema()))
define(Transaction.new(Table("DB.SCHEMA.TRANSACTIONS").to_schema()))
```

### Define Edges

Create the graph and define edges via field equality:

```python
gnn_graph = Graph(model, directed=True, weighted=False)
Edge = gnn_graph.Edge

define(Edge.new(src=Transaction, dst=Customer)).where(
    Transaction.customer_id == Customer.customer_id)
define(Edge.new(src=Transaction, dst=Article)).where(
    Transaction.article_id == Article.article_id)
```

#### Self-Referential Edges

When both sides of an edge are the same concept, use `.ref()`:

```python
PostRef = Post.ref()
define(Edge.new(src=Post, dst=PostRef)).where(
    PostRef.parent_id == Post.id)
```

When a separate relationship table mediates the self-reference:

```python
PeopleRef = People.ref()
define(Edge.new(src=People, dst=PeopleRef)).where(
    People.Id == Related.person1,
    PeopleRef.Id == Related.person2,
)
```

#### Multiple Typed Edges Between Same Pair

```python
BB1Edge = Concept("BB1Edge", extends=[Edge])
BB2Edge = Concept("BB2Edge", extends=[Edge])

Bref = B.ref()
define(BB1Edge.new(src=B, dst=Bref)).where(B.field1 == Bref.id)
define(BB2Edge.new(src=B, dst=Bref)).where(B.field2 == Bref.id)
```

---

## Phase 3: Task Tables & Relationships

**Goal:** Define train/val/test task tables and the relationship templates that encode the task structure.

**What to gather from the user:**
- Snowflake locations for train/val/test tables (may differ from source tables — use the task table location from Phase 1)
- Join key back to the source concept (e.g., `nct_id` to `Study`)
- Label/target column name (e.g., `did_not_finish`, `outcome`)
- Whether there's a time column (e.g., `timestamp`)
- For link prediction: the target concept and its join key

### Task Table Concepts

Task table concepts have no `identify_by`:

```python
train_table_concept = Concept("TrainTable")
val_table_concept = Concept("ValidationTable")
test_table_concept = Concept("TestTable")

define(train_table_concept.new(Table("DB.TASKS.TRAIN").to_schema()))
define(val_table_concept.new(Table("DB.TASKS.VAL").to_schema()))
define(test_table_concept.new(Table("DB.TASKS.TEST").to_schema()))
```

### Relationship Templates

Relationships encode the task structure using a template string with three parts:
- **Head** = source concept (the concept being predicted on)
- **"at" clause** = optional timestamp field
- **"has" clause** = label (classification/regression) or target concept (link prediction)

**Relationship arity rules:**

| Task Type | Train/Val template | Test template |
|-----------|-------------------|---------------|
| classification (no time) | `f"{Source} has {Any:label}"` | `f"{Source}"` |
| classification (with time) | `f"{Source} at {Any:ts} has {Any:label}"` | `f"{Source} at {Any:ts}"` |
| regression (no time) | `f"{Source} has {Any:value}"` | `f"{Source}"` |
| regression (with time) | `f"{Source} at {Any:ts} has {Any:value}"` | `f"{Source} at {Any:ts}"` |
| link_prediction | `f"{Source} has {Target}"` | `f"{Source}"` |
| repeated_link_prediction | `f"{Source} at {Any:ts} has {Target}"` | `f"{Source} at {Any:ts}"` |

### Node Classification (with time)

```python
Train = Relationship(f"{User} at {Any:timestamp} has {Any:target}")
define(Train(User, train_table_concept.timestamp, train_table_concept.target)).where(
    User.user_id == train_table_concept.user
)

Val = Relationship(f"{User} at {Any:timestamp} has {Any:target}")
define(Val(User, val_table_concept.timestamp, val_table_concept.target)).where(
    User.user_id == val_table_concept.user
)

Test = Relationship(f"{User} at {Any:timestamp}")
define(Test(User, test_table_concept.timestamp)).where(
    User.user_id == test_table_concept.user
)
```

### Node Classification (no time)

```python
Train = Relationship(f"{User} has {Any:target}")
Val = Relationship(f"{User} has {Any:target}")
Test = Relationship(f"{User}")
```

### Link Prediction (with time / repeated_link_prediction)

```python
Train = Relationship(f"{Customer} at {Any:timestamp} has {Article}")
define(Train(Customer, train_table_concept.timestamp, Article)).where(
    Customer.customer_id == train_table_concept.customer_id,
    Article.article_id == train_table_concept.article_id,
)

Val = Relationship(f"{Customer} at {Any:timestamp} has {Article}")
define(Val(Customer, val_table_concept.timestamp, Article)).where(
    Customer.customer_id == val_table_concept.customer_id,
    Article.article_id == val_table_concept.article_id,
)

Test = Relationship(f"{Customer} at {Any:timestamp}")
define(Test(Customer, test_table_concept.timestamp)).where(
    Customer.customer_id == test_table_concept.customer_id,
)
```

### Link Prediction (no time)

```python
Train = Relationship(f"{Customer} has {Article}")
Val = Relationship(f"{Customer} has {Article}")
Test = Relationship(f"{Customer}")
```

---

## Phase 4: Problem Definition

**Goal:** Now that data and task tables are understood, confirm the task type and evaluation metric.

**Decisions to make:**
- **Task type**: Follows from Phase 3 — if the label is binary → `binary_classification`, if numeric → `regression`, if target is a concept → `link_prediction` or `repeated_link_prediction` (with time)
- **Eval metric**: Choose based on task type (see table below)
- **Time column**: If relationships use "at" keyword → set `has_time_column=True`

**Default metrics per task type:**

| Task Type | Suggested Metric |
|-----------|-----------------|
| binary_classification | `roc_auc` |
| multiclass_classification | `accuracy` |
| multilabel_classification | `multilabel_auprc_macro` |
| regression | `rmse` |
| link_prediction | `link_prediction_precision@5` |
| repeated_link_prediction | `link_prediction_precision@5` |

For all valid (task_type, eval_metric) combinations, see [references/task-types-and-metrics.md](references/task-types-and-metrics.md).

---

## Phase 5: Feature Configuration

**Goal:** Configure PropertyTransformer to annotate concept fields with their semantic types.

```python
pt = PropertyTransformer(
    category=[User.locale, User.gender, Event.city, Event.state, Event.country],
    datetime=[User.joinedAt, Event.start_time],
    continuous=[User.birthyear],
    time_col=[Event.start_time],
)
```

### Feature Type Guidelines

| Data type | Annotation |
|-----------|-----------|
| Boolean flags, enum/status codes | `category` |
| Ages, prices, ratings | `continuous` |
| Free-form text, names, descriptions | `text` |
| Dates, timestamps | `datetime` |
| Explicit integer values (not IDs) | `integer` |

### Feature Selection Strategy

- **Drop all PKs and FKs.** Graph structure already captures relationships; IDs add noise. Example: `drop=[Study.nct_id, Outcome.id, Outcome.nct_id, ...]`
- **Start with minimal `text` fields.** Text embedding is expensive and too many text fields dilute signal. Begin with 3–5 key text fields (titles, identifiers like `mesh_term`), add more only if metrics improve.
- **Use `category` for discrete location/status fields.** Fields like city, state, country have limited cardinality — `category` encodes them more efficiently than `text`.
- **Use `continuous` for numeric measurements.** Counts, scores, percentages — anything with meaningful numeric ordering.
- **Lean feature sets beat everything-in.** In practice, reducing ~30 text fields to 5 improved AUROC from 57% to 68%. When in doubt, drop first and add later.

PropertyTransformer is optional — omitting it auto-infers all field types. For production, explicit annotation is recommended. Use `drop` to exclude fields or entire concepts: `drop=[Customer, Article.COLOUR_GROUP_CODE]`.

For the full feature type reference including drop patterns, see [references/property-transformer-types.md](references/property-transformer-types.md).

---

## Phase 6: Training

**Goal:** Configure the GNN estimator with hyperparameters and run training.

### GNN Constructor

#### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `database`, `schema` | Snowflake location of source data tables (from Phase 1) |
| `exp_database`, `exp_schema` | Snowflake location for experiment artifacts (from Phase 1) |
| `graph` | Graph object with edges defined (from Phase 2) |
| `train`, `validation` | Relationship objects (from Phase 3) |
| `task_type` | From Phase 4 |
| `eval_metric` | From Phase 4 |

#### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pt` | None | PropertyTransformer (from Phase 5; omit for auto-inference) |
| `has_time_column` | False | Set `True` when Relationships use the "at" keyword |
| `stream_logs` | True | Stream training logs to console |

### Node Classification Example

```python
gnn = GNN(
    database="MY_DB", schema="MY_SCHEMA",
    exp_database="MY_DB", exp_schema="EXPERIMENTS",
    graph=gnn_graph, pt=pt,
    train=Train, validation=Val,
    task_type="binary_classification",
    eval_metric="roc_auc",
    device="cuda", n_epochs=5, lr=0.005,
)
gnn.fit()
```

### Link Prediction Example (temporal)

```python
gnn = GNN(
    database="MY_DB", schema="MY_SCHEMA",
    exp_database="MY_DB", exp_schema="EXPERIMENTS",
    graph=gnn_graph, pt=pt,
    train=Train, validation=Val,
    task_type="repeated_link_prediction",
    eval_metric="link_prediction_precision@5",
    has_time_column=True,
    device="cuda", n_epochs=5, lr=0.005,
    head_layers=2, num_negative=20, label_smoothing=True,
)
gnn.fit()
```

### Common Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` (GPU) or `"cpu"` |
| `n_epochs` | 5 | Number of training epochs |
| `lr` | 0.005 | Learning rate |
| `train_batch_size` | 256 | Training batch size |

For link prediction, also consider: `head_layers=2`, `num_negative=20`, `label_smoothing=True`.

For all hyperparameters, see [references/hyperparameters.md](references/hyperparameters.md). For tuning guidance when results are poor, see the "Tuning When Results Are Poor" section in that reference.

### Detecting `has_time_column`

If the Train Relationship template contains the "at" keyword (e.g. `f"{User} at {Any:timestamp} ..."`), set `has_time_column=True`.

---

## Phase 7: Evaluation & Debugging

**Goal:** Inspect what the engine received, check prediction results, and tune if results are poor.

### Inspecting the Dataset

After `gnn.fit()`, inspect what data the engine received:

```python
# Visual graph of the dataset schema (requires pydot)
graph_viz = gnn.dataset.visualize_dataset(show_dtypes=True)
graph_viz.write_png("dataset_schema.png")

# Export full metadata as a dictionary (useful for debugging feature types)
config = gnn.dataset.metadata_dict

# Print the data config to console
gnn.dataset.print_data_config()
```

### Predictions

After training, generate predictions on the test set:

```python
Source.predictions = gnn.predictions(domain=Test)
```

#### Classification (binary, multiclass, multilabel)

```python
User.predictions = gnn.predictions(domain=Test)

select(
    User.user_id,
    User.predictions.probs,
    User.predictions.predicted_labels,
).where(User.predictions).inspect()
```

#### Regression

```python
Unit.predictions = gnn.predictions(domain=Test)

select(
    Unit.unit_id,
    Unit.predictions.predicted_value,
).where(Unit.predictions).inspect()
```

#### Link Prediction

```python
Customer.predictions = gnn.predictions(domain=Test)

select(
    Customer.customer_id,
    Article.article_id,
    Customer.predictions.rank,
    Customer.predictions.scores,
).where(
    Customer.predictions.predicted_article == Article,
).inspect()
```

The `predicted_<target>` attribute name is derived from the target concept name, always lowercase:
- Target `Article` -> `.predicted_article`
- Target `Product` -> `.predicted_product`

#### As DataFrame

Replace `.inspect()` with `.to_df()` to get a pandas DataFrame:

```python
df = select(
    User.user_id,
    User.predictions.probs,
    User.predictions.predicted_labels,
).where(User.predictions).to_df()
```

**Prediction attributes by task type:**

| Task Type | Attributes |
|-----------|-----------|
| classification | `.probs`, `.predicted_labels` |
| regression | `.predicted_value` |
| link prediction | `.rank`, `.scores`, `.predicted_<target>` |

For the full prediction attributes reference, see [references/prediction-attributes.md](references/prediction-attributes.md).

### Tuning Poor Results

If results are significantly worse than expected, check these in order:
1. **Inspect the dataset** — verify feature types and edges match expectations
2. **Reduce text features** — too many text fields dilute signal (see Phase 5 Feature Selection Strategy)
3. **Adjust hyperparameters** — see [references/hyperparameters.md](references/hyperparameters.md) "Tuning When Results Are Poor" section

---

## Phase 8: Model Management (optional)

**Goal:** Register a trained model for reuse, or load a previously trained model for inference.

### Register a Model

After `gnn.fit()` completes:

```python
gnn.register_model(
    model_database="MY_DB",
    model_schema="MODEL_REGISTRY",
    model_name="fraud_detector",
    version_name="v1",
    comment="Initial training run",  # optional
)
```

### Load a Model

#### By Registry Key

```python
gnn = GNN(
    database="MY_DB", schema="MY_SCHEMA",
    exp_database="MY_DB", exp_schema="EXPERIMENTS",
    graph=gnn_graph, property_transformer=pt,
    source_concept=User,
    task_type="binary_classification",
    model_database="MY_DB", model_schema="MODEL_REGISTRY",
    model_name="fraud_detector", version_name="v1",
)
gnn.load()
User.predictions = gnn.predictions(domain=Test)
```

#### By Run ID

```python
gnn = GNN(
    database="MY_DB", schema="MY_SCHEMA",
    exp_database="MY_DB", exp_schema="EXPERIMENTS",
    graph=gnn_graph, property_transformer=pt,
    source_concept=User,
    task_type="binary_classification",
    model_run_id="01c2d9a0-0711-c54d-000a-1dc707f7a1e6",
)
gnn.load()
User.predictions = gnn.predictions(domain=Test)
```

**What to include vs. omit when loading:**

| Include | Omit |
|---------|------|
| `database`, `schema` | `train`, `validation` |
| `exp_database`, `exp_schema` | `eval_metric` |
| `graph`, `property_transformer` | hyperparameters (`device`, `n_epochs`, etc.) |
| `source_concept` (required — the concept being predicted on) | |
| `task_type` (required — needed to determine prediction structure) | |
| `target_concept` (required for link prediction only) | |
| model identifier (registry key or run ID) | |

After `gnn.load()`, use `gnn.predictions(domain=Test)` exactly as after `gnn.fit()`.

### Train-Register-Load Workflow

#### Session 1: Train and Register (Phases 1–6 + 8)

```python
gnn = GNN(
    database="MY_DB", schema="MY_SCHEMA",
    exp_database="MY_DB", exp_schema="EXPERIMENTS",
    graph=gnn_graph, pt=pt,
    train=Train, validation=Val,
    task_type="binary_classification", eval_metric="roc_auc",
    has_time_column=True, device="cuda", n_epochs=5,
)
gnn.fit()
gnn.register_model(
    model_database="MY_DB", model_schema="MODEL_REGISTRY",
    model_name="fraud_detector", version_name="v1",
)
```

#### Session 2: Load and Predict (rebuild Phases 2+5, then load)

```python
# Rebuild graph and property_transformer (same structure as training)
gnn_graph = Graph(model, directed=True, weighted=False)
# ... define edges ...
pt = PropertyTransformer(...)

gnn = GNN(
    database="MY_DB", schema="MY_SCHEMA",
    exp_database="MY_DB", exp_schema="EXPERIMENTS",
    graph=gnn_graph, property_transformer=pt,
    source_concept=User,
    task_type="binary_classification",
    model_database="MY_DB", model_schema="MODEL_REGISTRY",
    model_name="fraud_detector", version_name="v1",
)
gnn.load()
User.predictions = gnn.predictions(domain=Test)
```

---

## Common Pitfalls

| Mistake | Cause | Fix |
|---------|-------|-----|
| Concept name is plural (e.g. "Customers") | Naming convention | Use singular names: `Concept("Customer")` |
| Task table concept has `identify_by` | Task tables don't need primary keys | Use plain `Concept("TrainTable")` with no `identify_by` |
| Snowflake table name not fully qualified | Missing database or schema prefix | Use `"DATABASE.SCHEMA.TABLE"` format |
| Test Relationship includes label/target | Test data should not contain the answer | Omit the "has" clause: `f"{Source}"` or `f"{Source} at {Any:ts}"` |
| Positional args in `define(Train(...))` don't match template | Template and population call must align | Match the order: source, [timestamp], [label/target] |
| Self-referential edge without `.ref()` | Same concept on both sides creates ambiguity | Use `PostRef = Post.ref()` for the destination |
| `time_col` fields not in `datetime` list | Both lists must include the field | Add time columns to both `datetime=[...]` and `time_col=[...]` |
| Task table concept used in edge definition | Only graph concepts participate in edges | Edges connect domain entities, not task tables |
| Missing type import | e.g. using `Date` without importing it | Add missing types to the import line |
| Column name has spaces or special characters (e.g. `weight(kg)`) | Python identifier rules prevent `Concept.weight(kg)` | Use `getattr(People, "weight(kg)")` to reference the field |
| Passing `select(...)` fragments to `train=` or `validation=` | GNN expects Relationship objects | Use `train=Train` with the Relationship object directly |
| Missing `has_time_column=True` | Relationships use "at" keyword but flag not set | Set `has_time_column=True` when templates contain "at" |
| Using `.predicted_Article` (uppercase) | Attribute name is always lowercase | Use `.predicted_article` regardless of concept casing |
| Invalid task_type/metric combination | Not all metrics work with all task types | Check [references/task-types-and-metrics.md](references/task-types-and-metrics.md) |
| Calling `register_model()` before `fit()` | Model must be trained first | Always call `gnn.fit()` before `gnn.register_model()` |
| Omitting `graph` or `property_transformer` when loading | Loaded models still need graph structure | Provide the same `graph` and `property_transformer` used during training |
| Passing training-only params when loading | `train`, `validation`, hyperparameters are not needed | Omit them when calling `gnn.load()` |
| Omitting `source_concept` when loading | Load workflow requires `source_concept` explicitly — it cannot be inferred without `train=` | Add `source_concept=<YourConcept>` to the load GNN constructor |
| Omitting `task_type` when loading | Load workflow requires `task_type` to determine prediction structure | Add `task_type="<your_task_type>"` to the load GNN constructor |
| Omitting `target_concept` for link prediction load | Link prediction load workflow also requires `target_concept` | Add `target_concept=<YourTargetConcept>` when loading a link prediction model |

---

## Examples

| Pattern | Description | File |
|---------|-------------|------|
| Node classification (modeling) | Binary classification data model (User/Event/EventAttendee) | [examples/node_classification_snowflake.py](examples/node_classification_snowflake.py) |
| Link prediction (modeling) | Repeated link prediction data model (Customer/Article/Transaction) | [examples/link_prediction_snowflake.py](examples/link_prediction_snowflake.py) |
| Node classification (training) | Binary classification training + prediction | [examples/train_node_classification.py](examples/train_node_classification.py) |
| Link prediction (training) | Repeated link prediction training + prediction | [examples/train_link_prediction.py](examples/train_link_prediction.py) |
| Register and load | Complete train-register-load workflow across sessions | [examples/register_and_load.py](examples/register_and_load.py) |

---

## Reference files

| Reference | Description | File |
|-----------|-------------|------|
| PropertyTransformer types | Full feature type reference, drop patterns, and guidelines | [references/property-transformer-types.md](references/property-transformer-types.md) |
| Task types and metrics | All valid (task_type, eval_metric) combinations | [references/task-types-and-metrics.md](references/task-types-and-metrics.md) |
| Hyperparameters | Full hyperparameter table with types, defaults, and descriptions | [references/hyperparameters.md](references/hyperparameters.md) |
| Prediction attributes | Prediction attributes by task type with usage examples | [references/prediction-attributes.md](references/prediction-attributes.md) |
