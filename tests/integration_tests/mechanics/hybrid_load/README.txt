This is an illustration of how to autoload information from HYBRID to HERON

This script auto-loads the needed economic information about any component from a HYBRID folder to a HERON input XML file.

It takes the following command line argument:
1. HERON_input file: the original HERON xml file (pre_heron_input.xml) before loading any information from HYRBID

To run the script, use the following command:
python hybrid2heron_economic.py path/to/HERON/xml/file

For example, the terminall command looks like this:
python hybrid2heron_economic.py pre_heron_input.xml

The initial HERON xml file must have
1- A "Components" node : <Components>
2- At least one "Component" sub-node under the "Components" node:
  <Component name="component_name">
3- An empty "economics" node under the "component" node with "src" attribute. 
For example: <economics src="path/to/HYBRID/file"></economics>
