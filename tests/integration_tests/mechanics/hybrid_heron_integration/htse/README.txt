This is an illustration of how to autoload information from HYBRID to HERON

This script auto-loads the needed economic information about all the components from a HYBRID folder to a HERON input XML file.

It takes the following command-line arguments:
1. HYBRID_input folder: the path to the HYBRID folder from which the information is loaded
2. HERON_input file: the original HERON XML file before loading any information from HYBRID

To run this script, use the following command:
python hybrid2heron.py path/to/HYRBID/folder path/to/HYRBID/xml/file

For example, the command looks like this:
python hybrid2heron.py Costs/ heron_input.xml
