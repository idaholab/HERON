import os
import yaml
import xml.etree.ElementTree as ET
import io



# xml_files = [f for f in os.listdir('.') if f.endswith('.xml')]
xml_files = ['C2N_project_definitions.xml', 'settings.xml', 'agent_specifications.xml', 'ALEAF_settings.xml', 'single_agent_testing.xml', 'unit_specs.xml']

def generate_scheduled_retirements_dict(scheduled_retirements_element):
    scheduled_retirements_dict = {}
    for unit_type_elem in scheduled_retirements_element:
        unit_type = unit_type_elem.tag
        retirement_pd_num_units_dict = {}
        for child_elem in unit_type_elem:
            if child_elem.tag == 'retirement_pd':
                retirement_pd_list = child_elem.text.split(',')
                retirement_pd_num_units_dict = dict(zip(retirement_pd_list, [0] * len(retirement_pd_list)))
            elif child_elem.tag == 'num_units':
                num_units_list = child_elem.text.split(',')
                retirement_pd_num_units_dict = dict(zip(retirement_pd_num_units_dict.keys(), num_units_list))
        scheduled_retirements_dict[unit_type] = retirement_pd_num_units_dict
    return scheduled_retirements_dict    

def generate_dict(root):
    data = {}
    for child in root:
        data[child.tag] = {}
        for sub_child in child:
            tag = sub_child.tag
            text = sub_child.text
            # if sub child has sub children, call get_sub_data
            if len(sub_child) > 0:
                # if sub sub child has sub children, call get_sub_data
                if len(sub_child[0]) > 0:
                    if sub_child.tag == 'scheduled_retirements':
                        data[child.tag][tag] = generate_scheduled_retirements_dict(sub_child)
                        continue
                    else:
                        sub_data = {}
                        for sub_sub_child in sub_child:
                            sub_data[sub_sub_child.tag] = get_sub_data(sub_sub_child.tag, sub_sub_child, yaml.safe_load)
                        data[child.tag][tag] = sub_data
                else:
                    data[child.tag][tag] = get_sub_data(tag, sub_child, yaml.safe_load)
            elif tag == 'allowed_xtr_types':
                alowed_xtr = text.split(', ')
                data[child.tag][tag] = alowed_xtr    
            else:
                data[child.tag][tag] = text
    return data


def get_sub_data(tag, child, converter=str):
    sub_data = {}
    for sub_child in child:
        if sub_child.tag == 'eligibility':
            sub_data[sub_child.tag] = {}
            for sub_sub_child in sub_child:
                sub_data[sub_child.tag][sub_sub_child.tag] = sub_sub_child.text.split(', ')
        else:
            sub_data[sub_child.tag] = converter(sub_child.text)

    return sub_data


for xml_file in xml_files:
    filename = xml_file.split('.')[0]
    # load XML data from file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yaml_data = generate_dict(root)
    # pprint(yaml_data)
    yaml_lines = []
    for key, value in yaml_data.items():
        # if key starts with '_', remove it
        if key[0] == '_':
            key = key[1:]
        yaml_lines.append(f'{key}: ')
        for k, v in value.items(): 
            # add indentation for all the subkeys 
            stream = io.StringIO()
            # if v is a dict, add indentation for all the subkeys in v
            yaml.dump({k: v}, stream, indent=2) 
            # if stream has multiple lines, add indentation for all the lines, if it contains quotes, remove them
            if len(stream.getvalue().splitlines()) > 1:
                for line in stream.getvalue().splitlines():
                    line = '    ' + line
                    # if line.strip()[-1] is a quote, remove it
                    if line.strip()[-1] == "'":
                        yaml_lines.append(line.replace("'", ""))
                    elif line.strip()[-1] == '"':
                        yaml_lines.append(line.replace('"', ""))
                    else:
                        yaml_lines.append(line)
            else:
                # if stream.getvalue().strip() have quotes for the last word, remove them
                if stream.getvalue().strip()[-1] == '"':
                    yaml_lines.append('  ' + stream.getvalue().strip().replace('"', ""))
                elif stream.getvalue().strip()[-1] == "'":
                    yaml_lines.append('  ' + stream.getvalue().strip().replace("'", ""))
                else:
                    yaml_lines.append('  ' + stream.getvalue().strip())
        yaml_lines.append('\n')
    # for all the line in yaml_lines that start with '-', add indentation
    for i, line in enumerate(yaml_lines):
        if line.strip().startswith('-'):
            line = '  ' + line
            yaml_lines[i] = line


    with open(filename + '.yml', 'w') as f:
        f.write('\n'.join(yaml_lines))
        # f.write(yaml.dump(yaml_data, default_flow_style=False))
# end of for loop
import shutil
# create a folder to store the yaml files
if not os.path.exists('yml'):
    os.makedirs('yml')
# move the yaml files to the yml folder
for f in os.listdir('.'):
    if f.endswith('.yml'):
        shutil.move(f, 'yml')



