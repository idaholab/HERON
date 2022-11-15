import xml.etree.ElementTree as ET
import copy

def increment(item,d):
  item = item.strip().rsplit('_',1)[0]+'_{}'.format(d)
  return item

def modifyInput(root,mod_dict):
  Samplers = root.find('Samplers')
  # mc
  mc = Samplers.find('MonteCarlo')
  if mc is not None:
    # get the amount of denoising
    denoises = mod_dict['Samplers|MonteCarlo@name:mc_arma_dispatch|constant@name:denoises']
    capacities = {}
    comps = list(x.split(':')[-1].rsplit('_',1)[0] for x in mod_dict.keys() if '_capacity' in x)
    for comp in comps:
      capacities[comp] = mod_dict['Samplers|MonteCarlo@name:mc_arma_dispatch|constant@name:{}_capacity'.format(comp)]

    mc.find('samplerInit').find('limit').text = str(denoises)
    for comp, cap in capacities.items():
      for const in mc.findall('constant'):
        if const.attrib['name'] == comp+'_capacity':
          const.text = str(cap)
          break
  return root
