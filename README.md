# ts-transform-matherika-internal
Internal repo for TSFM project


## Code organization philosophy

1. Install all code --> never change it (until some distinct update event)
2. Download and clean all the data --> never change it (until some distinct update event)
3. Run the simulation --> change very often ("Research")
4. Run the analysis --> change not so often ("compare results")

## Meeting 08 May 2025
* Prediction over few hours
* Limited scope for discretionary trading
* Mid term future
	* Models that make a difference
* Being attractive to talent
* Questions
	* Which are the right questions?  <-- workshop
	* **When** to buy? 
	* Market impact?
* How to create value?
	* Order execution strategy
	* Risk management for market timing
	* Ask client to allow for a price margin
* Onboarding?

## New Research Questions

- Do TSFMs capture technical analysis on stocks?


## Updated new research questions (Oct 2025)
- What exactly is the benefit of a longer context?
	- Draw a curve: context size on x-axis, precision measure (eg RSME) on y-axis
	- Do we need **even longer** context to predict longer horizons? --> Draw two lines on previous curve, one for "short term" prediction, one for "medium term"
- Look ito Morai MoE
- If temp has no impact on returns with Chronos, try multiplying all retunrns by 100 (i.e. in %) ... this may help the tokenizer --> whole new set of research questions
- Are there more hallucinations in the case of (almost) constant data in the context?