
import xml.etree.ElementTree as ET
import io
import yaml

# load XML data from file
tree = ET.parse('abce.xml')
root = tree.getroot()

# Convert to dictionary
data = {}
for child in root:
    data[child.tag] = {}
    for sub_child in child:
        data[child.tag][sub_child.tag] = sub_child.text
        if sub_child.tag == 'allowed_xtr_types':
            data[child.tag][sub_child.tag] = yaml.safe_load(sub_child.text)
        elif sub_child.tag == 'policies':
            sub_sub_data = {}
            for sub_sub_child in sub_child:
                sub_sub_sub_data = {}
                for sub_sub_sub_child in sub_sub_child:
                    if  sub_sub_sub_child.tag == 'eligible':
                        sub_sub_sub_data[sub_sub_sub_child.tag] = yaml.safe_load(sub_sub_sub_child.text)
                    else:
                        sub_sub_sub_data[sub_sub_sub_child.tag] = sub_sub_sub_child.text
                sub_sub_data[sub_sub_child.tag] = sub_sub_sub_data
            data[child.tag][sub_child.tag] = sub_sub_data


# add comment lines and convert to YAML
yaml_lines = []
for key, value in data.items():
    yaml_lines.append('#'*56)
    #add key with same length as comment lines and centered in comment lines
    yaml_lines.append('#' + ' '*(27-len(key)//2) + key + ' '*(27-len(key)//2) + '#')
    yaml_lines.append('#'*56)
    yaml_lines.append('')
    yaml_lines.append(f'{key}: ')
    if key == 'file_paths' or key == 'ALEAF':
        for k, v in value.items():
            # add quotes to file paths
            yaml_lines.append(f'  {k}: "{v}"')
    else:
        for k, v in value.items():
            stream = io.StringIO()
            if k == 'allowed_xtr_types':
                yaml.dump({k: v}, stream, )
                yaml_lines.append('  '+stream.getvalue().strip().replace("- ", "    - "))
            else:
                yaml.dump({k: v}, stream)
                # if stream.getvalue().strip() have quotes for the last word, remove them
                if stream.getvalue().strip()[-1] == "'":
                    yaml_lines.append('  '+stream.getvalue().strip().replace("'", ""))
                else:
                    yaml_lines.append('  '+stream.getvalue().strip())
    
    yaml_lines.append('')

# write YAML data to file
with open('settings.yml', 'w') as f:
    f.write('\n'.join(yaml_lines))


