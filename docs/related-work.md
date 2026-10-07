# Related work

Corollary is not a new theory. It combines well-established ideas from knowledge representation and
incremental computation, and applies them to LLM agents. This page credits that work and explains where
Corollary differs from tools you may already use.

## Foundations

**Truth maintenance systems.** Jon Doyle's *A Truth Maintenance System* (1979) introduced the
justification-based TMS: beliefs labeled `IN` or `OUT` from recorded justifications, with
non-monotonic "out-lists" and dependency-directed backtracking. Corollary's kernel is a JTMS in this
tradition: well-founded labeling, `unless` (out-list) justifications, and retraction cascades.

**Assumption-based TMS.** Johan de Kleer's ATMS (1986) labels each belief with the sets of assumptions
under which it holds, which lets a reasoner explore alternatives in parallel. Corollary doesn't implement
an ATMS yet. It is on the roadmap for exploring competing hypotheses.

**Belief revision.** The AGM theory (Alchourrón, Gärdenfors and Makinson, 1985) formalizes how a rational
agent should revise its beliefs when it learns something that contradicts them. Corollary's conflicts and
resolvers are a practical, policy-driven take on the same question: when sources disagree, which belief
gives way, and on what grounds.

**Why TMSs didn't spread.** Classic truth maintenance required every justification to be written by hand
or derived from a hand-built rule base. That was the bottleneck. Language models can now propose claims
and justifications at scale, while the TMS keeps them consistent. That pairing is the bet behind Corollary.

## Incremental computation

**Spreadsheets and build systems** (Make, Bazel, Salsa, Adapton) recompute only what depends on a changed
input, and stop when a recomputed result is unchanged (*early cutoff*). Corollary applies the same
discipline to an agent's conclusions. A useful one-line description of the project is *incremental
recomputation for agent reasoning*. The difference is that some of Corollary's "build steps" are model
calls, so their dependencies must be inferred conservatively rather than declared exactly.

**Data provenance and lineage**, in databases and data pipelines, records where each value came from.
Corollary records provenance too, and also uses it: the provenance graph drives retraction, repair and
verification.

## Agent memory and frameworks

**Message-log agent frameworks** store state as a growing transcript. They are simple and flexible, but a
wrong fact remains in the log. Corollary keeps a log only as a view (`kb.transcript()`) and builds every
model context from the belief base instead.

**Memory layers and temporal knowledge graphs** for agents store facts or summaries, often with
timestamps, and some can invalidate facts that are contradicted or outdated. Zep's Graphiti, for
example, tracks the validity intervals of facts in a temporal knowledge graph. Corollary approaches this
from the perspective of epistemic justification: rather than just expiring facts by time or temporal updates,
Corollary tracks justifications and dependencies so that when a premise is invalidated, all conclusions
derived from it are automatically retracted via dependency-directed backtracking.

**Hierarchical memory (MemGPT and Letta).** MemGPT and its successor Letta borrow operating system
virtual memory concepts to let agents manage their own context (Packer et al., 2023). They page information
back and forth between a limited working context and unbounded archival or recall storage.
- *How Corollary relates:* MemGPT/Letta decides *what stays in context* based on capacity and recency/relevance.
  Corollary decides *what is still true* based on logical and evidentiary justification. A paged-out fact in
  MemGPT is still considered valid in its storage layer; whereas in Corollary, when a premise is retracted,
  its consequences are invalidated regardless of whether they are in active working memory or archival storage.
  The two are orthogonal: an agent could use Letta for managing its working context window while using
  Corollary as its underlying belief and truth maintenance store.

**Graph orchestration frameworks (LangGraph and similar).** LangGraph and similar frameworks define an
agent's control flow as a graph of steps with typed shared state, supporting checkpoints after every step,
time travel, and human-in-the-loop interrupts (Chase et al., 2023).
- *How Corollary relates:* LangGraph orchestrates *what runs when* and manages control flow graphs and state
  snapshots. Corollary tracks *what is believed and why*. While LangGraph's checkpoints can rewind and re-run
  an execution path, they do not record fine-grained semantic dependencies between individual derived claims;
  thus, a downstream correction cannot be selectively propagated across arbitrary derivations without re-running
  the graph nodes. The two are highly complementary: a Corollary belief base can serve as the structured,
  consistent state object maintained across LangGraph nodes.

## References

- Alchourrón, C. E., Gärdenfors, P., & Makinson, D. (1985). On the logic of theory change: Partial meet contraction functions and their associated revision functions. *Journal of Symbolic Logic*, 50(2), 510-530.
- Chase, H., et al. (2023). LangGraph: Multi-actor applications with LLMs. LangChain. [https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- de Kleer, J. (1986). An assumption-based TMS. *Artificial Intelligence*, 28(2), 127-162.
- Doyle, J. (1979). A truth maintenance system. *Artificial Intelligence*, 12(3), 231-272.
- Packer, C., Fang, V., Patil, S., Lin, K., Wooders, S., & Gonzalez, J. E. (2023). MemGPT: Towards OS-inspired LLM memory management. *arXiv preprint arXiv:2310.08560*. [https://letta.com](https://letta.com)
