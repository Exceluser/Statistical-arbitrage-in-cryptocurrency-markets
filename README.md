# Statistical-arbitrage-in-cryptocurrency-markets
Repository to show and share parts of the code used for creating the results explored in the paper "Statistical arbitrage in cryptocurrency markets". Short description of the code per file in the following.

**crypto_forest_example:** 

Showcase of the complete "walk-through" of the project from loading and building the data, defining features, splitting the data, building and training the model, and performing the backtest. Used model is a random forest. Relies on the helper functions and classes defined in the following files.

**knn_crypto_script:**

Analogeous to _crypto_forest_example_, but with a different model (knn instead of random forest). No backtest in this file.

**crypto_dataprovider:**

Helper class defined to load and build the data from seperate csv-files or from a consolidated hdf file.

**feature_generator:**

Helper functions from calculating features used as input for the predictive models.

**kpi_backtest:**

Helper functions to calculate the kpis in the backtest.

**helper:**

Helper class to split training and test data in the desired manner. Helper function to perform backtests.
