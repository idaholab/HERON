This test demonstrated converting the HERON optimized components' variables (optimized dispatch) to a text file that is compatible with HYBRID via two steps:

Step #1: "create_user_input.py"
This script is run to create a user-input file for the HYBRID user.
It is an initial step that the HYBRID user needs to create new names, if necessary, to: 1- The optimized dispatch outputs, 2- The components' capacities.
This script extracts the optimized dispatch outputs from the dispatch prints csv file and the list of components (and their capacities) from the HERON input XML file.

It takes the following arguments:
1- The HERON input XML file
2- The optimized dispatch outputs csv file.
For example, To run the script, use the following command:
    python create_user_input.py heron_input.xml dispatch_print.csv
The output will be a user input file: "user_input.txt".
The user needs to review the user input file and change/review the HYRBID variables and capacities there.

Next, the user can run the "export2Hybrid.py" which loads the dispatch outpus from HERON to a new file that HYBRID can use. The variables names in the autoloaded HYBRID file are borrowd from "user_input.txt"


Step #2: "export2Hybrid.py"
A script to convert the HERON optimized components' variables (optimized dispatch) to a text file that is compatible with HYBRID.

The user-input file, "user_input.txt", must be in the same folder.
The user must review/modify the "user_input.txt" before running this script.

This script is run with the following arguments:
1- The HERON input XML file.
2- The optimized dispatch outputs CSV file.
For example, to run this script, use the following command:
    python export2Hybrid.py heron_input.xml dispatch_print.csv
The output will be a text file compatible with HYBRID: "hybrid_input.txt."
