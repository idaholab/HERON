#!/usr/bin/env python
# Copyright 2024, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
This script will automatically convert old and incompatible HERON XML
input files to the latest XML specification or most recent HERON version.

WARNING: This script removes comments from XML files, if your comments are
important to you, please make sure you write your output to a new file as
demonstrated below.

How to use:

`./h1_to_h2.py "your_input_file.xml"`

This will automatically create a new input file "new_your_input_file.xml" while
keeping your old file untouched!

`./h1_to_h2.py "your_input_file.xml" --dry-run`

This will pipe the new XML structure to your terminal stdout!

`./h1_to_h2.py "your_input_file.xml" -o "new_custom_name.xml"`

This will output a newly named XML file. If you wish to overwrite the previous
input file simply provide the same name to the the `-o` argument.
"""
import xml.etree.ElementTree as ET
import argparse
import os

def transform_xml(input_file, output_file, dry_run) -> None:
    try:
        # Parse the input XML file
        tree = ET.parse(input_file)
        root = tree.getroot()

        # Find the Components section
        components_section = root.find('Components')

        if components_section is not None:
            # Iterate over all the Component elements
            for component in components_section.findall('Component'):
                # Handle the economics element
                economics = component.find('economics')
                if economics is not None:
                    # Handle the lifetime element
                    lifetime = economics.find('lifetime')
                    if lifetime is not None:
                        lifetime_value = lifetime.text
                        economics.set('lifetime', lifetime_value)
                        economics.remove(lifetime)

                    # Handle the CashFlow elements within economics
                    cash_flows = economics.findall('CashFlow')
                    for cash_flow in cash_flows:
                        # Handle the depreciate element
                        depreciate = cash_flow.find('depreciate')
                        if depreciate is not None:
                            depreciate_value = depreciate.text
                            cash_flow.set('depreciate', depreciate_value)
                            cash_flow.remove(depreciate)

                        # Remove mult_target attribute if it exists
                        if 'mult_target' in cash_flow.attrib:
                            del cash_flow.attrib['mult_target']

                # Handle the produces element
                produces = component.find('produces')
                if produces is not None:
                    # Handle the consumes element
                    consumes = produces.find('consumes')
                    if consumes is not None:
                        consumes_value = consumes.text
                        produces.set('consumes', consumes_value)
                        produces.remove(consumes)

                    # Handle the ramp_limit element
                    ramp_limit = produces.find('ramp_limit')
                    if ramp_limit is not None:
                        ramp_limit_value = ramp_limit.text
                        produces.set('ramp_limit', ramp_limit_value)
                        produces.remove(ramp_limit)

                    # Handle the ramp_freq element
                    ramp_freq = produces.find('ramp_freq')
                    if ramp_freq is not None:
                        ramp_freq_value = ramp_freq.text
                        produces.set('ramp_freq', ramp_freq_value)
                        produces.remove(ramp_freq)

                # Handle the stores element
                stores = component.find('stores')
                if stores is not None:
                    # Handle the periodic_level element
                    periodic_level = stores.find('periodic_level')
                    if periodic_level is not None:
                        periodic_level_value = periodic_level.text
                        stores.set('periodic_level', periodic_level_value)
                        stores.remove(periodic_level)

                    # Handle max_charge_rate
                    max_charge_rate = stores.find('max_charge_rate')
                    if max_charge_rate is not None:
                        max_charge_rate_value = max_charge_rate.text
                        stores.set('max_charge_rate', max_charge_rate_value)
                        stores.remove(max_charge_rate)

                    # Handle max_discharge_rate
                    max_discharge_rate = stores.find('max_discharge_rate')
                    if max_discharge_rate is not None:
                        max_discharge_rate_value = max_discharge_rate.text
                        stores.set('max_discharge_rate', max_discharge_rate_value)
                        stores.remove(max_discharge_rate)

                    # Handle the RTE element
                    rte = stores.find('RTE')
                    if rte is not None:
                        rte_value = rte.text
                        stores.set('rte', rte_value)
                        stores.remove(rte)

        if dry_run:
            # Print the modified XML to standard output for dry-run
            ET.dump(tree)
        else:
            # Write the modified XML to the output file
            tree.write(output_file, xml_declaration=True, encoding='utf-8')
            print(f"Changes written to {output_file}")

    except ET.ParseError as e:
        print(f"Error parsing XML file '{input_file}': {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Transform XML files to fix backwards compatibility errors.")
    parser.add_argument("input_file", help="Path to the input XML file.")
    parser.add_argument("-o", "--output-file", help="Path to the output XML file. Defaults to 'new_{input_file_name}'.")
    parser.add_argument("--dry-run", action="store_true", help="Show the changes that would be made without writing to a file.")

    args = parser.parse_args()

    # Default the output file name if not provided
    if args.output_file:
        output_file = args.output_file
    else:
        input_filename = os.path.basename(args.input_file)
        output_file = os.path.join(os.path.dirname(args.input_file), f"new_{input_filename}")

    transform_xml(args.input_file, output_file, args.dry_run)

if __name__ == "__main__":
    main()