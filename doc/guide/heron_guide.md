
# Performing HERON Analysis

As a part of research software analysis, using HERON to obtain meaningful information about long-term economic viability for IES systems can be a complex process. In this guide we provide some guidance to help simplify the process.

## Overview

To effectively use HERON, several pieces of information need to be gathered and prepared. The common activities preceeding a HERON analysis often include the following:

- Market Characterization
- Technology Identification
- Technology Characterization
- Time History Training
- Pre-analysis Calculations

Each of these is described in more detail below.

After gathering the requisite data, the process of running HERON itself may involve multiple steps and iteration to assure the model is performing as expected. In general, the following process may be used:

1. Assemble HERON input file
1. Run HERON in "debug" mode
    - Check that cash flows are reasonable
    - Check that energy dispatch is reasonable
1. Run coarse optimization with HERON
    - Check optimization path
    - Check behavior near optimal solution
1. Run production-level optimization
1. Run "debug" mode at optimal solution
    - Check cash flows, energy dispatch

Each of these steps is described in more detail below.

## Before Running HERON

In this section we describe some of the common activities that are often required to perform useful HERON analysis.

#### Market Characterization
The market(s) in which an IES will be evaluated have a strong impact on HERON analysis. Markets include both electricity markets as well as other commodity markets.

- Electricity markets

Determining the characteristics and structure of the electricity markets to consider can be an involved process. Electricity markets are usually split into "regulated" markets, where generation units are owned and operated by a single entity and the goal is to minimize cost to meet demand; and "deregulated" markets, in which competing generation bids into a centralized market and profitability for individual generation units is the goal.

If considering a deregulated market, a decision must be made whether to model the IES as a "price taker" or "price maker". In "price taker" approaches, the IES passively observes electricity prices and can choose reactively how much energy to provide to the market without feedback on the price of electricity. In the "price maker" approach, there is an expected feedback between the IES operation and the price of electricity. Subjectively, if the IES operational range is sufficently small that electricity prices do not change notably based on IES dispatch, then a "price taker" approach may be valid; otherwise, a "price maker" approach is required.

HERON natively supports regulated markets as well as "price taker" deregulated markets. "Price maker" approaches currently require custom dispatching. There are existing examples of this approach in the FORCE use cases repository.

Additionally, the electricity market may incentivize products beyond electricity generation, such as reserve capacity, spinning reserves, frequency control, and many others. Identifying the requirements, depth, and prices for these markets will allow them to be included in the HERON model.

- Other Commodity Markets

Given the nature of IES, frequently we consider secondary commodities such as hydrogen, clean water, chemical products, etc. Generally, we consider the "demand" for these products to be constant from the IES, modeling a contractual agreement. This allows deriving a constant price via supply/demand curves, which must be identified before running HERON.

#### Technology Identification

Identifying which technologies are going to be included in the HERON analysis is key to successful performance. The number of technologies required is strongly dependent on the market characterization; a regulated market will require most of the technologies present to be simulated, whereas a price taker deregulated market often only requires the IES components to be modeled. All the technologies that are required to evaluate the performance of the IES in the chosen market should be identified and represented in HERON for accurate analysis.

For each technology, it must be decided whether the technology can be dispatched flexibly ("independent" or "dependent") or whether it has a fixed dispatch. Further, it must be decided whether

#### Technology Characterization

For each technology identified above, several characterizing technical and economic properties must be obtained.

For production components (power generators, chemical processes, commodity generators) and storage components (batteries, reservoirs, tanks), the following characteristics are required:
- Technical Properties
  - Reasonable capacity installation limits (i.e., nameplace capacity)
  - Minimum operational limits (e.g., power level)
  - Transfer function (ratios between resources consumed and resources produced)
  - Ramping limitations (time to transition between operational levels)
  - Flexibility (does the component need to run at constant rate or can it be dispatched)

Note that for storage components, "capacity" is defined in terms of quantity (e.g., kg or MWh), while for production components, "capacity" is defined as rates (e.g., kg/s or MW).

- Economic Properties
  - Capital costs (i.e., CAPEX), including economy of scale factor
  - Fixed operational costs (i.e., FOM), which are invariant with respect to how the component is dispatched
  - Variable operational costs (i.e., VOM), which are dependent on dispatch
  - Fuel costs, when the fuel is not a modeled resource in the system
  - Incentives (e.g., policy-driven incentives for producing some resources)
  - Component effective lifetime
  - Amortization schedule

In addition, general economic factors such as the firm's discout rate, inflation, tax rates, and so on need to be determiend.

#### Time History Training

One of the notable features of HERON is analysis of expected economic efficacy including uncertainty. Most of this uncertainty comes through synthetic histories representing energy demand, electricity prices, solar and wind availability, etc. For each of these uncertain histories, appropriate synthetic histories must be acquired from existing trained models or trained for the analysis using RAVEN.

Note that static histories can be used for a lower-order approximation of expected economic efficacy; however, for most analyses the uncertainty in histories should be considered in production-level runs.

The process of training and validating synthetic histories can be a significant effort. For some analysis, this process may represent up to half of the total effort required to accurately perform HERON analysis.

Note also that currently HERON does not include market and demand evolution. If such structure is desired, many capacity expansion models exist that can provide trends on which multiyear synthetic histories can be trained.

#### Pre-analysis Calculations

Before launching HERON, it can be productive to consider some "back of the envelope" calculations to develop an expectation for understanding how the HERON analysis may perform. For example, using an average price of electricity compared to price of hydrogen might give insight on the expected profitable size of hydrogen contracts. Observing the daily variance of electricity prices may give indications of the potential profitability of building an energy storage system. A surprising amount of intuition can often be generated about a HERON run before ever starting the actual analysis using the information gathered and comparing to previous published analyses.


## Performing HERON Analysis

Once the information above has been gathered, a HERON analysis can be prepared and carried out.

#### Assemble HERON input file

All the gathered information can be assembled to create a HERON input file. While it is possible to generate one from scratch, typically finding a similar analysis from the regression tests or FORCE use cases is advisable.

#### Run HERON in "debug" mode

Once the HERON input has been assembled, it is critical to observe the assembled model and assure it performs as expected. To that end, HERON has a "debug" mode that allows a small subset of the overall problem to run, with more verbose output that can be checked in detail.

Debug runs are designed to be computationally light enough to run in minutes on a desktop computer.

- Check Energy Dispatch

The first behavior to check after a debug run is the energy dispatch. HERON provides automated plots for several different synthetic scenarios showing the dispatch of production, demands, and storage components in the model. Care should be taken to analyze behaviors in these examples critically for expected behavior.

The importance of this step can be illustrated with the true story of an early HERON use case. One feature of storage components such as batteries is round-trip efficiency, or RTE. This measures the lossiness of storage as a ratio between how much of a resource can be withdrawn to how much was deposited.

In a particular scenario, a large number of negative energy prices were observed due to over-installation of generating components. Given a fixed source of energy, the dispatch optimization elected to deposit and withdraw large amounts of energy from a storage unit, effectively using the RTE to "burn off" excess energy without being charged the negative prices at the market. While mathematically optimal, this solution was not intuitively expected by the analyst, and indicated the model as designed in the HERON input needed to be adjusted to see expected performance. While this behavior highlighted a fundamental challenge to energy generation in this scenario, the analyst would have been disappointed with a full analysis without considering the dispatch behavior.

- Check CashFlow Breakdown

HERON follows sign conventions to track costs and revenues to the firm within the system. It also uses the convention that when resources are consumed by a component they are negative values, while produced resources are positive. This allows for clear conservation of resources and checking behavior.

However, this also means that care must be taken when setting up cash flows in the HERON input so that the sign of cash flows is correct in the cost-revenue balance. For example, resources consumed at a demand market have a negative sign, but are generally associated with a sales revenue, so should have the sign flipped.

Checking the cash flows in a debug run to assure all the signs and total cash flows are behaving as intended is highly recommended to assure production-level analysis does not contain surprise incorrect behavior.

#### Run coarse optimization with HERON

While optional, running a high-tolerance optimization with HERON can allow the analyst to observe the behavior of the optimizer, check with their intuition, and adjust optimization parameters. Because "black box optimization" is extraordinarily sensitive to initial conditions, this can greatly reduce computational time in a long-running production analysis.

If the optimization path is proceeding in a direction that is not intuitive to the analyst, running debug runs at a couple points along the optimization path may give indication as to the driving motivations for the current optimization. These motivations may uncover need for adjustments in the input file to accurately capture the system or may simply reveal a correct but unintiuitive optimal solution.

The computational burden of a coarse optimization varies, but should run in minutes to hours on a high performance computing environment, or one or more days on a desktop computer.

#### Run production-level optimization

Once an analyst is satisfied with the debug and coarse optimizations, a full-scale low-tolerance HERON analysis can be launched.

With full parallel computing in place, a full optimization with roughly 100 scenario samples per capacity point and allowing roughly 600 optimization iterations to converge can take several hours on a high performance computing environment.

Without parallel computing, a similar analysis may take many days to converge, but could be run on a desktop computer.

After the optimization is complete, it is still common for analysts or project leads to request additional runs with modified model parameters or adjusted optimization tuning. As such, it is important to plan the first full HERON run be complete with plenty of time for iteration before final results are reported.

#### Run "debug" mode at optimal solution

Once an optimal point is found, it is typical for project leads to want cash flow and dispatch plots for example scenarios in addition to final optimization values. To this end, HERON's debug mode allows for specifying debug values to use, which will provide example dispatch plots and cash flow breakdowns for the optimal capacity point over the requested number of scenarios.

Note that in a typical HERON analysis, on the order of two million dispatch optimizations might be performed, which makes storing all the data used along the way impractical. In addition, the stochastic nature of time series mean that we can show dispatch plots for _example scenarios_, but these are only indicative of expected performance. The metrics calculated by HERON are based on thousands of these dispatches in a typical analysis. However, a few scenarios can be illustrative of typical behaviors.


## Advanced Features

### Custom User Specified Functions

HERON allows users to create their own functions that perform computations during simulation runtime. 

Currently, these functions can only deal with computations that do not occur during the dispatch optimization. For example, a user can write a function that determines the `<reference_price>` parameter of a component's cashflow because cashflows are not computed during the inner dispatch optimization. 

Currently, a user would _not_ be able to write a custom transfer function that informs the dispatcher on how resources are transformed while moving between components of the specified system. This is because transfer functions are required during the dispatch of the system and would require the user to write the function in a way that could be interpreted by our underlying optimization library. To be more specific, a user would **not** be able to use a custom function within a `<transfer>` XML node in the HERON input file. **While this feature is not currently available, it may be made available in the future.**

Users can see examples of these custom functions in the [FORCE use case repository.](https://github.com/idaholab/FORCE/tree/main/use_cases)

#### Custom Function API

Users can write custom functions, but they must follow the API conventions to ensure they work properly during runtime.

A custom function utilized in a HERON input file requires two input parameters that are always returned by the function:

* `data`: A Python dictionary containing information related to associated component that is calling the function.
* `meta`: A Python dictionary containing information pertaining to the case as a whole. 

It is possible to specify ancillary functions in the python file that do not follow the API conventions, but understand that functions called from the HERON input file will require this specification. 

For example, suppose a user wanted to write a function that computed the reference price for a particular component based the current year of the project. In the input file, under the `<reference_price>` node, the user would write:


```xml
...
<reference_price>
    <Function method="get_price">functions</Function>
</reference_price>
...

<!-- Also make sure to specify down in <DataGenerators> the path to the python file -->
<DataGenerators>
    <Function name="functions">[path/to/python/functions/file]</Function>
</DataGenerators>
```

Then in a file created by the user, they might write the following function:

```python
def get_price(data, meta):
    """Determine the time-dependent price multiplier and return new reference_price."""
    # For the first ten years of the project we can sell for a higher price
    year = meta['HERON']['active_index']['year']
    if year <=10:
        multiplier = 3
    else: 
        multiplier = 1.5
    result = 1000 * multiplier
    return {"reference_price": result}, meta
```

In the above code block, the function starts by accessing data from the `meta` parameter to determine what the current year is within the simulation. Then the function determines the multiplier based on the current year of the simulation. If the simulation is within the first ten years of the project timeline, then it sets a higher multiplier, otherwise it sets the multiplier lower. Finally, the function stores the newly computed `reference_price` into a dictionary that is returned by the function. This value will then be used as the `<reference_price>` within the component that this function is called from within the input file.  






### Custom User Specified Dispatch

**Under Construction**