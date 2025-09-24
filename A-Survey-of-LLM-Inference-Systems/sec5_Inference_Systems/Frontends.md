# **5.1 Frontends**

A **frontend** provides programmatic access to an LLM, giving users control over inputs and outputs while abstracting away some complexities of prompt engineering.



### **1. Capabilities of Frontends**

Frontends enable:

* **Prompt transformation**

  * User instructions → **Chain of Thought (CoT)**, **few-shot prompts**, or **RAG (Retrieval-Augmented Generation)** prompts.
* **Adaptive prompting**

  * Dynamically selecting prompts based on outputs of earlier ones.
* **Output constraints**

  * Enforcing **templates**, **formats**, or structured output (e.g., JSON).
* **Interleaved prompting**

  * Combines structured outputs + template completion.
  * Helps **speed up inference** by reducing unnecessary generations.



### **2. Prompting Strategies**

* Rich literature exists.
* Key idea: frontends integrate prompting best practices **into reusable modules**.



### **3. Interface Types**

* **Imperative Interfaces**

  * Lower-level APIs for building **LLM programs**.
  * Fine-grained control, but more complex to use.
* **Declarative Interfaces**

  * Higher-level, query-based.
  * Easier for users (like writing SQL queries or natural-language queries).



### **4. Table: Example Frontends & Features**

| **Frontend**      | **Interface** | **Control Flow** | **Structured Outputs** | **Template Completion** | **Prompt Optimization** | **Coupled Run** |
| ----------------- | ------------- | ---------------- | ---------------------- | ----------------------- | ----------------------- | --------------- |
| **LMQL**          | Declarative   | ✓                | ✓                      |     Stag.                    |                         |                 |
| **DSPy**          | Declarative   | ✓                | ✓                      |    Stream                     |         ✓                |                 |
| **SGLang**        | Imperative    |       ✓           | ✓                      | Stag.                       |                         |  ✓               |
| **Guidance**      | Imperative    | ✓                | ✓                      | Stream                       |                         |                 |
| **LangChain**     | Imperative    | ✓                | ✓                      | Stream                       |                         |                 |


**Legend**

* **Ctrl. Flow** = Can manage multi-step execution (control flow).
* **Struct. Out.** = Can enforce structured outputs.
* **Temp. Comp.** = Supports template completion.
* **Prompt Opt.** = Optimizes prompts automatically.
* **Coupled-Run.** = Can couple multiple prompts/runs together.



**In summary**:
Frontends make LLMs more programmable and efficient. Declarative ones (e.g., LMQL, DSPy) are user-friendly, while imperative ones (e.g., LangChain, Guidance) give more control. Features like **structured outputs, control flow, and interleaved prompting** enable faster, more reliable inference.


---


## **Basic Interaction**

### **1. Control Flow with Captured Outputs**

* LLM outputs can be **captured into variables** and then used like program values.
* Example: The LLM decides which tool to use (`calc` or `www`), and based on the captured value, the control flow branches.

#### Pseudocode Walkthrough

```python
s += assistant("To answer " + q + ", I need " + gen("tool", choices=["calc", "www"]))

if s["tool"] == "calc":
    # route execution to calculator
elif s["tool"] == "www":
    # route execution to web search
```

* `gen("tool", choices=[...])` constrains the LLM to only output one of the valid options (`calc` or `www`).
* The captured value (`s["tool"]`) then controls the downstream logic — like an `if...else` in normal programming.



### **2. Constrained Generation**

* This prevents the LLM from generating **unbounded responses** by imposing termination conditions.
* Useful for things like limiting response size, enforcing format, or stopping at a keyword.

#### Example (LMQL Declarative Syntax)

```lmql
"Give your answer here: [y]"
where len(TOKENS(y)) < 25
```

* `[y]` → captures the LLM’s answer.
* `len(TOKENS(y)) < 25` → stops generation once the answer exceeds 25 tokens.
* Declarative style means you can express constraints directly in the **prompt specification**, without extra imperative code.



#### **Key Takeaway**

* **Control flow (if/else, loops)** = achieved by capturing outputs and branching programmatically.
* **Constrained generation** = keeps LLM outputs bounded, predictable, and compliant with user-specified limits.

---


## **Structured Outputs**


### **1. What are Structured Outputs?**

* LLMs normally generate **free-form text**.
* Structured outputs constrain the generation to a **user-specified type/format**:

  * e.g., two-digit number, date, JSON schema, boolean, etc.
* Ensures outputs are **predictable and machine-usable**.



### **2. How Constraints are Enforced**

Frontends can implement structured outputs via several mechanisms:

1. **Prompt phrasing** → “Give answer as a two-digit number.”
2. **Logit masking** → prevents invalid tokens from being selected at decode time.

   * e.g., masking everything except tokens `00–99` when expecting a two-digit number.
3. **Token resampling** → if an invalid token is generated, resample until valid.
4. **Reprompting** → if the LLM produces an invalid result, ask again with refined instructions.



### **3. Template Completion**

* Extends structured outputs to fill **complex templates**, like JSON schemas.
* Two strategies:

  1. **Item-by-item prompting**

     * Prompt for each field separately (`name`, then `age`, then `top_songs`).
     * Pros: higher reliability, supports **interleaving prefill and decode** (faster inference with persisted caches).
  2. **Single-shot prompting**

     * Ask LLM to output the full template at once.
     * Cons: may need **multiple retries** until it strictly obeys the format.



### **4. Example – Constrained Template Generation**

#### Template Definition (LMQL program):

```json
{
  "name": "[STRING_VALUE]",
  "age": [INT_VALUE],
  "top_songs": [
    "[STRING_VALUE]",
    "[STRING_VALUE]"
  ]
}
```

* Bracketed items (`[STRING_VALUE]`, `[INT_VALUE]`) = **type constraints**.
* LMQL parses this into prompts, enforces constraints, and captures valid output.



### **5. Execution Workflow (Prefill & Decode Interleaving)**

1. **First prompt**

   ```text
   Write a summary of Bruno Mars, the singer:
   { "name": "
   ```

   → LLM outputs: `Bruno Mars`.

2. **Second prompt (reuse prior output)**

   ```text
   Write a summary of Bruno Mars, the singer:
   { "name": "Bruno Mars",
     "age": "
   ```

   → LLM outputs: `38`.

3. Repeat for `top_songs`.

Because the **prefix of each prompt overlaps**, the runtime can **reuse KV cache entries** (instead of recomputing from scratch). This reduces **prefill cost** and speeds up inference.

**Key Insight**: Structured outputs + template completion ensure **valid, parsable outputs** while also leveraging **runtime caching** for efficiency.


---

## **Declarative Frontends**

### **1. What are Declarative Frontends?**

* A **declarative interface** lets users specify **what they want**, not **how to execute it**.

* Instead of writing step-by-step procedural code (imperative), users **declare structures, constraints, or modules**, and the frontend handles prompt construction, chaining, and enforcement.

* Advantage: **higher-level abstraction** → easier to write, optimize, and reuse.



### **2. Examples of Declarative Frontends**

#### **(1) LMQL**

* Syntax inspired by **query languages** (SQL-like).

* Workflow:

  * Write a **prompt statement**.
  * Add a **from clause** → which model to use.
  * Add a **where clause** → constraints on output.

* LMQL **translates this declarative query** into a **prompt chain** and ensures output constraints are met.

* Since structure is **declared upfront**, LMQL can:

  * Treat the query as a **template**.
  * Use **prompt + decode interleaving** to leverage KV cache for faster inference.

  Example:

```lmql
"Translate the sentence into French: {answer}" 
from gpt3 
where len(answer) < 20
```

* `from` specifies the model.

* `where` enforces a constraint (max length).

* LMQL handles execution, caching, and constraint enforcement.



#### **(2) DSPy**

* **Object-oriented declarative interface**.

* Instead of writing query-like prompts, users **define modules** (like building blocks).

* Each module encapsulates a **prompting strategy**:

  * Few-shot prompting
  * Chain-of-thought reasoning
  * RAG retrieval

* Workflow:

  * User **declares a module**.
  * Provides **parameters** (prompt text, examples, constraints).
  * Module is **invoked like a function**.

 Benefit: DSPy can **automatically optimize prompts** — for example:

* Synthesizing few-shot examples.
* Adjusting instructions for higher accuracy.



### **3. Key Differences (LMQL vs DSPy)**

| Feature             | **LMQL**                                                  | **DSPy**                                                 |
| ------------------- | --------------------------------------------------------- | -------------------------------------------------------- |
| Style               | Query language (SQL-like)                                 | Object-oriented modules                                  |
| Input form          | Prompt + clauses                                          | Module declarations                                      |
| Constraint handling | `where` clauses enforce constraints                       | Modules encapsulate strategies & enforce rules           |
| Prompt optimization | No explicit optimization                                  | Built-in optimization (e.g., generate few-shot examples) |
| Efficiency          | Template-based → supports cache-aware decode interleaving | Module reuse + optimized prompting                       |


**Takeaway**:

* **LMQL**: Best for **structured queries + constraints** → "What do I want, under what conditions?"
* **DSPy**: Best for **modular and optimized prompting workflows** → "Which strategies should be applied, and with what parameters?"


---


## **Imperative Frontends**

Unlike declarative ones (where you describe *what* you want), **imperative frontends** give users **low-level, step-by-step control** over *how* prompts are constructed, executed, and post-processed.

Users write **procedural code** (e.g., Python) that directly manipulates prompts, outputs, and control flow. This allows **more flexibility**, but requires **more manual coding**.



### **Examples of Imperative Frontends**

#### **1. SGLang**

* **Low-level API** → direct interaction with LLM.
* **Speculative execution**:

  * LLM continues decoding *past the termination condition*.
  * If useful tokens are generated → avoids extra API calls.
  * Example: if user asked for a short sentence, the LLM may generate extra context beyond the stop token, saving another call.
* **Runtime integration**:

  * SGLang’s runtime supports **cache persistence**.
  * Enables **prompt + decode interleaving** for efficient template generation.

**Best for performance-oriented fine-grained control.**



#### **2. Guidance**

* Similar features to **LMQL**, but in **imperative syntax** instead of declarative.
* Example: Instead of SQL-like `where` constraints, you call a Python function like:

  ```python
  gen("answer", max_length=25, choices=["yes", "no"])
  ```
* Constraints are passed **as arguments to generation functions**.
* Feels natural for programmers who prefer imperative scripting.

**Best for structured outputs with code-first flexibility.**



#### **3. LangChain**

* More **object-oriented** than Guidance.
* Encapsulates **prompts** and **LLM outputs** inside **classes**.
* Allows **programmatic manipulation** of prompt objects.
* Provides **LangGraph (agent-based framework)**:

  * Lets you build **complex workflows** (nested if/else, tool use, multi-agent systems).
  * Example: an agent reasoning about whether to call a calculator, a search API, or another LLM.

**Best for building agent workflows and pipelines.**


### **Imperative vs Declarative (Summary Table)**

| Aspect        | **Declarative (LMQL, DSPy)**                          | **Imperative (SGLang, Guidance, LangChain)**        |
| ------------- | ----------------------------------------------------- | --------------------------------------------------- |
| Style         | High-level "what" statements                          | Low-level "how" code                                |
| User control  | Minimal (framework manages execution & constraints)   | Full control of flow, constraints, and API calls    |
| Optimization  | Built-in (template optimization, prompt optimization) | User decides optimization (but some runtimes help)  |
| Example usage | `where` clauses in LMQL                               | `if/else` conditions in Guidance or LangChain       |
| Best for      | Fast prototyping, structured outputs                  | Complex logic, agent workflows, fine-grained tuning |


**Takeaway:**

* Use **Declarative** when you want **simplicity + auto-optimization**.
* Use **Imperative** when you want **control + custom workflows** (especially for agents, speculative decoding, or tool use).

---


