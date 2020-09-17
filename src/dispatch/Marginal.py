
import os
import sys
import time as time_mod
from functools import partial
from collections import defaultdict
from utils import InputData, InputTypes, graphStructure
import numpy as np
import pandas as pd
from collections import defaultdict
#import pyomo.environ as pyo
#from pyomo.opt import SolverStatus, TerminationCondition
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

from .Dispatcher import Dispatcher
from .DispatchState import DispatchState, NumpyState
try:
  import _utils as hutils
except (ModuleNotFoundError, ImportError):
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
  import _utils as hutils


class MARGINAL(Dispatcher):
  """
    Dispatches using rolling windows in Pyomo
  """
  naming_template = {'comp prod': '{comp}|{res}|prod',
                     'comp transfer': '{comp}|{res}|trans',
                     'comp max': '{comp}|{res}|max',
                     'comp ramp up': '{comp}|{res}|rampup',
                     'comp ramp down': '{comp}|{res}|rampdown',
                     'conservation': '{res}|consv',
                    }

  ### INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = InputData.parameterInputFactory('Dispatcher', ordered=False, baseNode=None)
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'MarginalDispatcher' # identifying name
    self._window_len = 24         # time window length to dispatch at a time # FIXME user input

  def read_input(self, specs):
    """
      Read in input specifications.
      @ In, specs, RAVEN InputData, specifications
      @ Out, None
    """
    print('DEBUGG specs:', specs)
    
  ### API
  def dispatch(self, case, components, sources, meta):
    """
      Performs dispatch.
      @ In, components, list, HERON components available to the dispatch
      @ Out, dispatch
    """
      
    #Path = self.graph_maker_extracter(components)
    self.add_source_sinks(components)
    
    #print("This is unique path", Path)
    #aa
    ### This function is the same as implemented in pyomo_dispatch
    print(meta['HERON'].keys())
    print("These are the keys",meta['HERON']['RAVEN_vars'])
    #aaa
    #print("These are the components", components)
    #aaa
    #print("These are the sources", sources)
    t_start, t_end, t_num = self.get_time_discr()
    resources = sorted(list(hutils.get_all_resources(components)))
    #print("These are the resources", resources)
    #aa
    time = np.linspace(t_start, t_end, t_num)
    dispatch = NumpyState()
    start_index = 0
    final_index = len(time)
    # TODO window overlap!  ( )[ ] -> (   [  )   ]
    while start_index < final_index:
      end_index = start_index + self._window_len
      if end_index > final_index:
        end_index = final_index # + 1?
      specific_time = time[start_index:end_index]
      print('DEBUGG starting window {} to {}'.format(start_index, end_index))
      start = time_mod.time()
      subdisp = self.dispatch_window(specific_time,
                                     case, components, sources,
                                     meta)
      end = time_mod.time()
      print('DEBUGG solve time: {} s'.format(end-start))
      
      print("This is subdisp", subdisp)
      aaa
    
      for comp in components:
        for res, values in subdisp[comp.name].items():
          dispatch.set_activity_vector(comp, res, start_index, end_index, values)
      start_index = end_index 
    return dispatch
  
  def dispatch_window(self, t,case, components, sources, meta):
    """
      Computes the marginal cashflow and dispatch
      @In, time, case, components, sources, meta
      @Out, result
    """
    Component_produce_consume = []
    ## Extract producers
    for comp in components:
      if comp.get_interaction().tag == 'produces':
        Component_produce_consume.append(comp)
    Normalized_coefficients = {}
    self.resource_index_map = meta['HERON']['resource_indexer']
    ## Extract transfer coefficients
    for comp in Component_produce_consume:
 
      transfer = comp.get_interaction().get_transfer()
      
      if transfer is not None:
        Normalized_coefficients[comp]=(transfer._coefficients)
    ## Base and perturbed objects
    numpy_state_Base = MarginalNumpy()
    numpy_state_Perturbed = MarginalNumpy() 
    Base_Cash = []
    Perturbed_Cash = []
    Marginal_Cash = []
    #print("This is META HERON", meta['HERON'].keys())
    #aaa
    ## Resource indexer and resource delta
    self.Base = meta['HERON']['resource_indexer']
    self.Perturbed = meta['HERON']['resource_indexer']#self.create_resources(Resources_list,Components)
    self.delta = 100
    ## Intialize base and perturbed state objects
    numpy_state_Base.initialize(Component_produce_consume,self.Base,t)
    numpy_state_Perturbed.initialize(Component_produce_consume,self.Perturbed,t)
    ## For each producers perturb the resources
    #print("These are the producers", Component_produce_consume)
    for i in range(0,len(Component_produce_consume)):#len(Component_produce_consume)):
      resource_list_inp = (Component_produce_consume[i]).get_inputs()#get_resources()#list(self.Base[self.ke[i]].keys())
      if len(resource_list_inp) == 0:
        resource_list_op = Component_produce_consume[i].get_outputs()
        
        for resource in resource_list_op:
          random_ = 0#np.random.randint(0,1)## TODO how to set nominal base state optimally
          numpy_state_Base.set_activity(Component_produce_consume[i],resource,t,random_)
          numpy_state_Perturbed.set_activity(Component_produce_consume[i], resource,t, random_+ self.delta)
          Base_cash = self._compute_cashflows([Component_produce_consume[i]],numpy_state_Base,t,meta)
          Perturbed_cash = self._compute_cashflows([Component_produce_consume[i]],numpy_state_Perturbed,t,meta)
          Base_Cash.append(Base_cash)
          Perturbed_Cash.append(Perturbed_cash)
          ##Since single input therefore only partial derivative is needed ##TODO suppose component has multiple inputs then implement full total derivative
          Marginal_Cash.append(abs(Base_cash-Perturbed_cash)/self.delta)
          #print("These are the cash",self._compute_cashflows([Component_produce_consume[i]],numpy_state_Base,t,meta)
          #      ,self._compute_cashflows([Component_produce_consume[i]],numpy_state_Perturbed,t,meta))
          
      if len(resource_list_inp)!=0:
        for resource in resource_list_inp:
          transfer = Component_produce_consume[i].get_interaction().get_transfer()
          #output  = Component_produce_consume[i].get_outputs()

          if transfer is not None:
            random_ = random_ * list(transfer._coefficients.values())[-1]
            numpy_state_Base.set_activity(Component_produce_consume[i],resource,t,random_)
            numpy_state_Perturbed.set_activity(Component_produce_consume[i], resource,t, random_+ self.delta)
            Base_cash = self._compute_cashflows([Component_produce_consume[i]],numpy_state_Base,t,meta)
            Perturbed_cash = self._compute_cashflows([Component_produce_consume[i]],numpy_state_Perturbed,t,meta)
            Base_Cash.append(Base_cash)
            Perturbed_Cash.append(Perturbed_cash)
            Marginal_Cash.append(abs(Base_cash-Perturbed_cash)/self.delta)
    resource_list_op = Component_produce_consume[0].get_outputs()  

    self.Total_Base = self._compute_cashflows(Component_produce_consume,numpy_state_Base,t,meta)
    Cashflow_Perturbed  = self.PartialChange_cashflow(numpy_state_Perturbed,numpy_state_Base, Component_produce_consume, meta,numpy_state_Base)#self._compute_cashflows(components,numpy_state_Perturbed,[0],meta)

    #print("This is META-3", meta['HERON'].keys())
    #print("This is Cashflow", Cashflow_Perturbed)
    #print("This is MC, BC, PC", BC, MC, PC)
    #aaa
    #aaa
    resource_index_map = meta['HERON']['resource_indexer']
    #print("This is the ",resource_index_map[comp], comp)
    ### TODO I don't know how Paul is extracting the chains--ask him what is that 
    ### graph theoretic algorithm he was mentioning
    ### TODO Based on the marginal cashflows for two or more chains create a Component list and then pass it to _retrieve_solution method
    result = self._retrieve_solution(Component_produce_consume, resource_index_map)
    #aaaa
    return result #Cashflow_Perturbed
  
  def graph_maker_extracter(self, components):
    
    graph = self.construct_graph(components)
    print("This is the original", graph)
    key = list(graph.keys())
    Graph_Shape = {}
    Graph_Shape_2 = {}
    for ke in key:
      temp = list(graph[ke].keys())
      #print("this is temp", temp)
      for ke2 in temp:
        #print("This is ke",ke2)
        if ke2 == 'produced by':
          temp_key = ke2
          #print("This is produced",graph[ke][ke2])
          #for i in range(0,len(graph[ke][ke2])):
            #print("This is it",graph[ke][ke2][i])
        if ke2 == 'consumed by':
          s#elf.True_False(graph,temp_key,ke,ke2)
          #Temp_list = [True if temp_key.get_inputs() == graph[ke][ke2][m].get_outputs()]
          for j in range(0,len(graph[ke][temp_key])):
            #### True and False ##
            
            Graph_Shape[graph[ke][temp_key][j].name] = [graph[ke][ke2][m].name for m in range(0,len(graph[ke][ke2]))]
            Graph_Shape_2[graph[ke][temp_key][j]] = [graph[ke][ke2][m] for m in range(0,len(graph[ke][ke2]))]
    #graphobj = graphStructure.graphObject()
    #graphobj.__init__(Graph_Shape)
    #unique_path = graphobj.findAllUniquePaths()
    #print("This is the constructed graph", Graph_Shape)
    #print("It is connected", graphobj.isConnectedNet())
    return Graph_Shape, Graph_Shape_2
  
  def construct_graph(self,components):
    """ construct dict of resources to what components either consume or produce that resource """
    res_info = {}
    # add components
    print('DEBUGG comps:', components)
    for comp in components:
      ins = comp.get_inputs()
      outs = comp.get_outputs()
      #print("These are the outs", outs, ins)
      for res in ins:
        if res not in res_info:
          res_info[res] = {'produced by': [], 'consumed by': [comp]}
        else:
          res_info[res]['consumed by'].append(comp)
        #if outs == set():
        #  res_info[res] = {'Sink' : [comp]}
          
      for res in outs:
        if res not in res_info:
          res_info[res] = {'produced by': [comp], 'consumed by': []}
        else:
          res_info[res]['produced by'].append(comp)
    # add sources --> no, because they should be made available through a component
    #self.get_sources_sinks(components)
    #self.add_source_sinks(components)
    # DEBUGG
    for res, info in res_info.items():
      print('RESOURCE "{}":'.format(res))
      print('    produced by:', list(x.name for x in info['produced by']))
      print('    consumed by:', list(x.name for x in info['consumed by']))
    # sanity checking
    for res, info in res_info.items():
      if not info['produced by']:
        raise IOError('Resource "{}" is not produced by any component or source!'.format(res))
      if not info['consumed by']:
        print('WARNING: resource "{}" is not consumed by any components!'.format(res))
    #print("This is the graph in function", res_info)
    #aaa
    return res_info
  
  def get_sources_sinks(self,components):
    
    source_sinks={}
    counter_source = 0
    counter_sink = 0
    for comp in components:
      ins = comp.get_inputs()
      outs = comp.get_outputs()
      
      if outs == set():
        source_sinks['sink'+str(counter_sink)] = comp.name
        counter_sink += 1
      if ins == set():
        source_sinks['source'+str(counter_source)] = comp.name
        counter_source += 1
    print("These are the source_sinks", source_sinks)
    return source_sinks
  
# def add_source_sinks(self, components):
#   graph = self.construct_graph(components)
#   source_sinks = self.get_sources_sinks(components)
  
  def add_source_sinks(self, components):
    graph2, graph4 = self.graph_maker_extracter(components)
    list_of_dict=[]
    list_of_dict.append(graph2)
    graph3 = graph2.copy()
    print("These are the unique", graph2)
    source_sinks = self.get_sources_sinks(components) 
    startnode = []
    stopnode = []
    for key, values in source_sinks.items():
      print("These are the key",key,values)  
      if key.startswith('sink'):
        stopnode.append(values)
        ### These sink values should be present in the list "items"
        for ke, vals in graph2.items():
          list_of_dict.append(self.getkeybyvalues(ke,vals, values, graph2,graph4))
      elif key.startswith('source'):
        startnode.append(values)

    print("This is startnode, stopnode", startnode, stopnode)
    dd = defaultdict(list)
    for d in list_of_dict:
      for key, value in d.items():
        print("This is the key", key)
        dd[key].append(value)
    print("This is the default dict", dd['electr_market'])
    
  
    dd2 = {}
    for key, value in dd.items():
      
      flat_list = [item for sublist in value for item in sublist]
      flat_list = self.unique(flat_list)
      
      dd2[key] = flat_list
    print("This is dd2",dd2)
    print("This is graph2, graph4",graph2, graph4)
  
  
      
    graphobj = graphStructure.graphObject()
    graphobj2 = graphStructure.graphObject()
    graphobj2.__init__(graph2)
    graphobj.__init__(dd2)
    unique_path = graphobj.findAllPaths(startnode[0],stopnode[1])
    unique_path_2 = graphobj.findAllUniquePaths()
    #findAllUniquePaths()
    print("This is unique_path", unique_path)
    print("This is unique_path_2",unique_path_2)
    print("This is a loop", graphobj.isALoop())
    print("This is a loop", graphobj2.isALoop())
    #final = {k: v for d in list_of_dict for k, v in d.items()}
    #print("This is the final", final)
        
        #for kes, vals in graph2.items():
          #if values == kes:
        #print("Found", values)
      #elif key.startswith('sink'):
        #print("Found", values)
 
    

  #  key = list(graph.keys())
    #aaa
    #print("It is connected", graphobj.isConnectedNet())

  def True_False(self,Dict,ke,Key, Value):
    print("This is the Dict, Key, Value", Dict, ke,Key, Value)
    aaa
    
  def unique(self,list1):
    unique_list = []
    
    for x in list1:
      if x not in unique_list:
        unique_list.append(x)
    return unique_list
  

  def getkeybyvalues(self,key_2, list_2, value,graph_2,graph_4) :
    
    y = list_2.copy()
    #list_of_dict = []
    graph_x = graph_2.copy()
    graph_y = graph_2.copy()
    #print("This is list", list_2, value)
    keys = list(graph_4.keys())
    
    for i,item in enumerate(list_2):
    
      if value == item:
        #print("This is item", item, value, key_2) #keys)
        y[y.index(value)] = key_2
        graph_x[value] = y
        #list_of_dict.append(graph_x)
        #graph_y.update(graph_x)
        #print("Inside", graph_x)
    
    #rint("This is graph_2", graph_2, graph_x, list_2)
    return graph_x
    
        

  def _retrieve_solution(self, components, resource_index_map):
    """
      Make solution look like the way it was done by him
      @ In, m, pyo.ConcreteModel, associated (solved) model
      @ Out, result, dict, {comp: {resource: [production], etc}, etc}
    """
    result = {} # {component: {resource: production}}
    for comp in components:
      result[comp.name] = {}
      for res, comp_r in resource_index_map[comp].items():
        tf = comp.get_interaction().get_transfer()
        if tf is not None:
          result[comp.name][res] = np.array([(tf._coefficients)[res]])
        elif tf is None:
          result[comp.name][res] = np.array([1])
    #print("Our result", result)
    return result
    
    


  
  def PartialChange_cashflow(self,stateobject,stateobject_B,components, meta,stateobject_P):
    
    """
      Compute partial cashflows
      @ In, stateobjects, components, meta
      @ Out, Change in cash flow
    
    """
    ###TODO delta is set to 0.1 here, this is the amount of 
    ### perturbations added to the syste. 
    component_list = components
    partialChange_Cash =[]
    #CF = []
    #TCF = []
    #for i in range(0,len(component_list)):
      #CASH = self._compute_cashflows([component_list[i]],stateobject,[0],meta)
      #CF.append(CASH)
      #TFB = self._compute_cashflows([component_list[i]],stateobject_B,[0],meta)
      #TCF.append(TFB)
      #resource_ = (component_list[i]).get_resources()
      #for resource in resource_:
        #print("This is the state",stateobject.get_activity(component_list[i], resource,[0]))
        #print("This is the state2",stateobject_P.get_activity(component_list[i], resource,[0]))
        #print("This is is the resource", component_list[i].get_resources())
        #aaa
        #self.get_perturbed( component_list[i],stateobject, component_list[i].get_resources(),
        #                  stateobject.get_activity(component_list[i], resource,[0]),self.delta,[0])
    CASH = self._compute_cashflows(component_list,stateobject,[0],meta)
    #TCF = np.array(TCF)
    #CF = np.array(CF)
    #print("This is TCF, CF", TCF, CF)
    partialChange_Cash.append((self.Total_Base-CASH)/self.delta)
    #print("This is partialChange", partialChange_Cash)
    #self._retrieve_solution(components)
    #aaa
    #print("This is case",self.Total_Base,CASH)
    return partialChange_Cash
  


  def get_perturbed(self, component_name, activity_perturbed, resource, base_state, delta, time):
    """
     A method to compute the perturbed state
     of the system
     @ In, component_name which is a string
     @ In, activity_perturbed which is an object of the NumpyState class
     @ In, resource name
     @ In, base_state, computed base state of the system
     @ In, delta, the amount of perturbation to be added
     """
    #print("This is the base state", base_state)
    #aa
    activity_perturbed.set_activity(component_name,resource.pop(),time,base_state-delta)


  def get_perturbed_back(self, component_name, activity_perturbed, resource, base_state, delta, time):
    """
     A method to compute the perturbed state
     of the system
     @ In, component_name which is a string
     @ In, activity_perturbed which is an object of the NumpyState class
     @ In, resource name
     @ In, base_state, computed base state of the system
     @ In, delta, the amount of perturbation to be added
     """
    activity_perturbed.set_activity(component_name,resource.pop(),time,base_state+delta)

  def create_resources(self, resource_list, components_list):

    components_list_internal = [comp.name for comp in components_list]

    self.Base_resources = {}
    self.Perturbed_resources = {}
    print(components_list_internal,resource_list)
    for i in range(0, len(components_list_internal)):
      self.Base_resources[components_list_internal[i]] = {}
      self.Perturbed_resources [components_list_internal[i]] = {}
      j = 0
      for resource in resource_list[i]:
      #for i in range(0,len(resource_list)):
        random_integer = 0#np.random.random_integers(10)
        self.Base_resources[components_list_internal[i]][resource] = j#random_integer
        #self.Perturbed_resources[comp]['resource'+str(i)] = #random_integer - 0.1 #perturbation
        self.Perturbed_resources[components_list_internal[i]][resource] = random_integer - 0.1
        j = j+1

    return self.Base_resources, self.Perturbed_resources
  




print("Computing Jacobian and Perturbed state")

class MarginalNumpy(NumpyState):
  pass


        



class MarginalState(DispatchState):
  def __init__(self):
    DispatchState.__init__(self)

  def create_resources(self, resource_list, components_list):

    components_list_internal = [comp.name for comp in components_list]

    self.Base_resources = {}
    self.Perturbed_resources = {}
    print(components_list_internal,resource_list)
    for i in range(0, len(components_list_internal)):
      self.Base_resources[components_list_internal[i]] = {}
      self.Perturbed_resources [components_list_internal[i]] = {}
      j = 0
      for resource in resource_list[i]:
      #for i in range(0,len(resource_list)):
        random_integer = np.random.random_integers(10)
        self.Base_resources[components_list_internal[i]][resource] = j#random_integer
        #self.Perturbed_resources[comp]['resource'+str(i)] = #random_integer - 0.1 #perturbation
        self.Perturbed_resources[components_list_internal[i]][resource] = j#random_integer - 0.1
        j = j+1

    return self.Base_resources, self.Perturbed_resources
#  def initialize(self):
#    self.Base_resources = {'Var_x':1.0,'Var_y':2.0,'Var_z':3.0}   
#    self.Perturbed_resources = {'Var_x':0.9, 'Var_y':1.7, 'Var_z':2.8}  
#    return (self.Base_resources, self.Perturbed_resources)
#  def initialize(self, components, resources_map, times ):
#    """
#      Set up dispatch state to hold data
#      @ In, components, list, HERON components to be stored
#      @ In, resources, list, string resources to be stored
#      @ In, time, list, float times to store
#      @ Out, None
#    """
#    self._components = components
#    self._resources = resources_map
    ##  Use these two base and Perturbed resources for each components
    #{'Var_x':1.0,'Var_y':2.0,'Var_z':3.0} 
    #{'Var_x':0.9, 'Var_y':1.7, 'Var_z':2.8}
#    self.Base, self.Perturbed = self.create_activity(resource_list, number_of_components)
#    self._times = times






  

    


  








