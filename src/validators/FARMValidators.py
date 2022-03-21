
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Example class for validators.
"""
import numpy as np
# import pickle as pk
import os
import sys
import xml.etree.ElementTree as ET
from _utils import get_heron_loc

from utils import InputData, InputTypes

from .Validator import Validator

class FARM_Beta(Validator):
  """
    A FARM SISO Validator for dispatch decisions.(Dirty Implementation)
    Accepts parameterized A,B,C,D matrices from external XML file, and validate
    the dispatch power (BOP, SES & TES, unit=MW), and
    the next stored energy level (TES, unit=MWh)

    Haoyu Wang, ANL-NSE, Jan 6, 2022
  """
  # ---------------------------------------------
  # INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = Validator.get_input_specs()
    specs.name = 'FARM_Beta'
    specs.description = r"""Feasible Actuator Range Modifier, which uses a single-input-single-output
        reference governor validator to adjust the power setpoints issued to the components at the
        beginning of each dispatch interval (usually an hour), to ensure the the operational constraints
        were not violated during the following dispatch interval."""

    component = InputData.parameterInputFactory('ComponentForFARM', ordered=False, baseNode=None,
        descr=r"""The component whose power setpoint will be adjusted by FARM. The user need
        to provide the statespace matrices and operational constraints concerning this component,
        and optionally provide the initial states.""")
    component.addParam('name',param_type=InputTypes.StringType, required=True,
        descr=r"""The name by which this component should be referred within HERON. It should match
        the component's name in \xmlNode{Components}.""")

    component.addSub(InputData.parameterInputFactory('MatricesFile',contentType=InputTypes.StringType,
        descr=r"""The path to the Statespace representation matrices file of this component. Either absolute path
        or path relative to HERON root (starts with %HERON%/)will work. The matrices file can be generated from
        RAVEN DMDc or other sources."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsUpper',contentType=InputTypes.InterpretedListType,
        descr=r"""The upper bounds for the output variables of this component. It should be a list of
        floating numbers or integers."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsLower',contentType=InputTypes.InterpretedListType,
        descr=r"""The lower bounds for the output variables of this component. It should be a list of
        floating numbers or integers."""))
    component.addSub(InputData.parameterInputFactory('InitialState',contentType=InputTypes.InterpretedListType,
        descr=r"""The initial system state vector of this component. It should be a list of
        floating numbers or integers. This subnode is OPTIONAL in the HERON input file, and FARM will
        provide a default initial system state vector if \xmlNode{InitialState} is not present."""))

    specs.addSub(component)

    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'BaseValidator'
    self._tolerance = 1.003e-6

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    self._unitInfo = {}
    for component in inputs.subparts:
      name = component.parameterValues['name']
      xInit=[]
      for farmEntry in component.subparts:
        if farmEntry.getName() == "MatricesFile":
          matFile = farmEntry.value
          if matFile.startswith('%HERON%'):
            # magic word for "relative to HERON root"
            heron_path = get_heron_loc()
            matFile = os.path.abspath(matFile.replace('%HERON%', heron_path))
        if farmEntry.getName() == "OpConstraintsUpper":
          UpperBound = farmEntry.value
        if farmEntry.getName() == "OpConstraintsLower":
          LowerBound = farmEntry.value
        if farmEntry.getName() == "InitialState":
          xInit = farmEntry.value
      self._unitInfo.update({name:{'MatrixFile':matFile,'Targets_Max':UpperBound,'Targets_Min':LowerBound,'XInit':xInit,'vp_hist':[],'y_hist':[]}})

  # ---------------------------------------------
  # API
  def validate(self, components, dispatch, times, meta):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, extra information pertaining to validation
      @ Out, errs, list, information about validation failures
    """
    # errs will be returned to dispatcher. errs contains all the validation errors calculated in below
    errs = [] # TODO best format for this?

    # get time interval
    Tr_Update_hrs = float(times[1]-times[0])
    Tr_Update_sec = Tr_Update_hrs*3600.

    # loop through the <Component> items in HERON
    for comp, info in dispatch._resources.items():
      # e.g. comp= <HERON Component "SES""> <HERON Component "SES"">
      # loop through the items defined in the __init__ function
      for unit in self._unitInfo:
        # e.g. CompInfo, unit= SES
        # Identify the profile as defined in the __init__ function
        if str(unit) not in str(comp):
          # If the "unit" and "comp" do not match, go to the next "unit" in loop
          continue
        else: # when the str(unit) is in the str(comp) (e.g. "SES" in "<HERON Component "SES"">")
          self._unitInfo[unit]['vp_hist']=[]; self._unitInfo[unit]['y_hist']=[]
          """ Read State Space XML file (generated by Raven parameterized DMDc) """
          MatrixFile = self._unitInfo[unit]['MatrixFile']
          Tss, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array = read_parameterized_XML(MatrixFile)

          """ MOAS steps Limit """
          g = int(Tr_Update_sec/Tss)+1 # numbers of steps to look forward, , type = <class 'int'>
          """ Keep only the profiles with YNorm within the [y_min, y_max] range """
          y_min = self._unitInfo[unit]['Targets_Min']
          y_max = self._unitInfo[unit]['Targets_Max']

          para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array = check_YNorm_within_Range(
            y_min, y_max, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array)

          if YNorm_list == []:
            print('ERROR:  No proper linearization point (YNorm) found in Matrix File. \n\tPlease provide a state space profile linearized within the [y_min, y_max] range\n')
            sys.exit('ERROR:  No proper linearization point (YNorm) found in Matrix File. \n\tPlease provide a state space profile linearized within the [y_min, y_max] range\n')
          max_eigA_id = eig_A_array.argmax()
          A_m = A_list[max_eigA_id]; B_m = B_list[max_eigA_id]; C_m = C_list[max_eigA_id]; D_m = np.zeros((p,m)) # all zero D matrix

          # loop through the resources in info (only one resource here - electricity)
          for res in info:
            if str(res) == "electricity":
              # loop through the time index (tidx) and time in "times"
              if self._unitInfo[unit]['XInit']==[]:
                x_sys = np.zeros(n)
              else:
                x_sys = np.asarray(self._unitInfo[unit]['XInit'])-XNorm_list[0]


              for tidx, time in enumerate(times):
                # Copy the system state variable
                x_KF = x_sys
                """ Get the r_value, original actuation value """
                current = float(dispatch.get_activity(comp, res, times[tidx]))
                # check if TES: power = (curr. MWh energy - prev. MWh energy)/interval Hrs

                if comp.get_interaction().is_type('Storage') and tidx == 0:
                  init_level = comp.get_interaction().get_initial_level(meta)

                if str(unit) == "TES":
                  # Initial_Level = float(self._unitInfo[unit]['Initial_Level'])
                  Initial_Level = float(init_level)
                  if tidx == 0: # for the first hour, use the initial level. charging yields to negative r_value
                    r_value = -(current - Initial_Level)/Tr_Update_hrs
                  else: # for the other hours
                    # r_value = -(current - float(dispatch.get_activity(comp, res, times[tidx-1])))/Tr_Update_hrs
                    r_value = -(current - Allowed_Level)/Tr_Update_hrs
                else: # when not TES,
                  r_value = current # measured in MW

                """ Find the correct profile according to r_value"""
                profile_id = (np.abs(para_array - r_value)).argmin()

                # Retrive the correct A, B, C matrices
                A_d = A_list[profile_id]; B_d = B_list[profile_id]; C_d = C_list[profile_id]; D_d = np.zeros((p,m)) # all zero D matrix
                # Retrive the correct y_0, r_0 and X
                y_0 = YNorm_list[profile_id]; r_0 = float(UNorm_list[profile_id])

                # Build the s, H and h for MOAS
                s = [] # type == <class 'list'>
                for i in range(0,p):
                  s.append([abs(y_max[i] - y_0[i])])
                  s.append([abs(y_0[i] - y_min[i])])

                H, h = fun_MOAS_noinf(A_d, B_d, C_d, D_d, s, g) # H and h, type = <class 'numpy.ndarray'>

                # first v_RG: consider the step "0" - step "g"
                v_RG = fun_RG_SISO(0, x_KF, r_value-r_0, H, h, p) # v_RG: type == <class 'numpy.ndarray'>

                """ 2nd adjustment """
                # MOAS for the steps "g+1" - step "2g"
                Hm, hm = fun_MOAS_noinf(A_m, B_m, C_m, D_m, s, g)
                # Calculate the max/min for v, ensuring the hm-Hxm*x(g+1) always positive for the next g steps.
                v_max, v_min = fun_2nd_gstep_calc(x_KF, Hm, hm, A_m, B_m, g)

                if v_RG < v_min:
                  v_RG = v_min
                elif v_RG > v_max:
                  v_RG = v_max

                # # Pretend there is no FARM intervention
                # v_RG = np.asarray(r_value-r_0).flatten()

                v_value = v_RG + r_0 # absolute value of electrical power (MW)
                v_value = float(v_value)

                # Update x_sys, and keep record in vp_hist and yp_hist within this hour
                for i in range(int(Tr_Update_sec/Tss)):
                  self._unitInfo[unit]['vp_hist'].append(v_value)
                  y_sim = np.dot(C_d,x_sys)
                  self._unitInfo[unit]['y_hist'].append(y_sim+y_0)
                  x_sys = np.dot(A_d,x_sys)+np.dot(B_d,v_RG)

                # Convert to V1:

                # if str(unit) == "TES":
                if comp.get_interaction().is_type('Storage'):
                  if tidx == 0: # for the first hour, use the initial level
                    Allowed_Level = Initial_Level - v_value*Tr_Update_hrs # Allowed_Level: predicted level due to v_value
                  else: # for the other hours
                    Allowed_Level = Allowed_Level - v_value*Tr_Update_hrs
                  V1 = Allowed_Level
                else: # when not TES,
                  V1 = v_value

                # print("Haoyu Debug, unit=",str(unit),", t=",time, ", curr= %.8g, V1= %.8g, delta=%.8g" %(current, V1, (V1-current)))

                # Write up any violation to the errs:
                if abs(current - V1) > self._tolerance*max(abs(current),abs(V1)):
                  # violation
                  errs.append({'msg': f'Reference Governor Violation',
                              'limit': V1,
                              'limit_type': 'lower' if (current < V1) else 'upper',
                              'component': comp,
                              'resource': res,
                              'time': time,
                              'time_index': tidx,
                              })

    # if errs == []: # if no validation error:
    #   print(" ")
    #   print("*********************************************************************")
    #   print("*** Haoyu Debug, Validation Success, Print for offline processing ***")
    #   print("*********************************************************************")
    #   print(" ")
    #   t_hist = np.arange(0,len(self._unitInfo['TES']['vp_hist'])*Tss,Tss)
    #   for unit in self._unitInfo:
    #     y_hist = np.array(self._unitInfo[unit]['y_hist']).T
    #     # print(str(unit),y_hist)
    #     for i in range(len(t_hist)):
    #       print(str(unit), ",t,",t_hist[i],",vp,",self._unitInfo[unit]['vp_hist'][i],",y1,",y_hist[0][i], ",y1min,",self._unitInfo[unit]['Targets_Min'][0],",y1max,",self._unitInfo[unit]['Targets_Max'][0],",y2,",y_hist[1][i], ",y2min,",self._unitInfo[unit]['Targets_Min'][1],",y2max,",self._unitInfo[unit]['Targets_Max'][1])


    return errs




def read_parameterized_XML(MatrixFileName):
  tree = ET.parse(MatrixFileName)
  root = tree.getroot()
  para_array = []; UNorm_list = []; XNorm_list = []; XLast_list = []; YNorm_list =[]
  A_Re_list = []; B_Re_list = []; C_Re_list = []; A_Im_list = []; B_Im_list = []; C_Im_list = []
  for child1 in root:
    for child2 in child1:
      for child3 in child2:
        if child3.tag == 'dmdTimeScale':
          Temp_txtlist = child3.text.split(' ')
          Temp_floatlist = [float(item) for item in Temp_txtlist]
          TimeScale = np.asarray(Temp_floatlist)
          TimeInterval = TimeScale[1]-TimeScale[0]
        if child3.tag == 'UNorm':
            for child4 in child3:
              para_array.append(float(child4.attrib['ActuatorParameter']))
              Temp_txtlist = child4.text.split(' ')
              Temp_floatlist = [float(item) for item in Temp_txtlist]
              UNorm_list.append(np.asarray(Temp_floatlist))
            para_array = np.asarray(para_array)

        if child3.tag == 'XNorm':
          for child4 in child3:
            Temp_txtlist = child4.text.split(' ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            XNorm_list.append(np.asarray(Temp_floatlist))

        if child3.tag == 'XLast':
          for child4 in child3:
            Temp_txtlist = child4.text.split(' ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            XLast_list.append(np.asarray(Temp_floatlist))

        if child3.tag == 'YNorm':
          for child4 in child3:
            Temp_txtlist = child4.text.split(' ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            YNorm_list.append(np.asarray(Temp_floatlist))

        for child4 in child3:
          for child5 in child4:
            if child5.tag == 'real':
              Temp_txtlist = child5.text.split(' ')
              Temp_floatlist = [float(item) for item in Temp_txtlist]
              if child3.tag == 'Atilde':
                A_Re_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Btilde':
                B_Re_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Ctilde':
                C_Re_list.append(np.asarray(Temp_floatlist))

            if child5.tag == 'imaginary':
              Temp_txtlist = child5.text.split(' ')
              Temp_floatlist = [float(item) for item in Temp_txtlist]
              if child3.tag == 'Atilde':
                A_Im_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Btilde':
                B_Im_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Ctilde':
                C_Im_list.append(np.asarray(Temp_floatlist))

  n = len(XNorm_list[0]) # dimension of x
  m = len(UNorm_list[0]) # dimension of u
  p = len(YNorm_list[0]) # dimension of y

  # Reshape the A, B, C lists
  for i in range(len(para_array)):
      A_Re_list[i]=np.reshape(A_Re_list[i],(n,n)).T
      A_Im_list[i]=np.reshape(A_Im_list[i],(n,n)).T
      B_Re_list[i]=np.reshape(B_Re_list[i],(m,n)).T
      B_Im_list[i]=np.reshape(B_Im_list[i],(m,n)).T
      C_Re_list[i]=np.reshape(C_Re_list[i],(n,p)).T
      C_Im_list[i]=np.reshape(C_Im_list[i],(n,p)).T

  A_list = A_Re_list
  B_list = B_Re_list
  C_list = C_Re_list

  eig_A_array=[]
  # eigenvalue of A
  for i in range(len(para_array)):
      w,v = np.linalg.eig(A_list[i])
      eig_A_array.append(abs(max(w)))
  eig_A_array = np.asarray(eig_A_array)

  return TimeInterval, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array

def check_YNorm_within_Range(y_min, y_max, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array):
  UNorm_list_ = []; XNorm_list_ = []; XLast_list_ = []; YNorm_list_ =[]
  A_list_ = []; B_list_ = []; C_list_ = []; para_array_ = []; eig_A_array_ =[]

  for i in range(len(YNorm_list)):
    state = True
    for j in range(len(YNorm_list[i])):
      if YNorm_list[i][j] < y_min[j] or YNorm_list[i][j] > y_max[j]:
        state = False
    if state == True:
      UNorm_list_.append(UNorm_list[i])
      XNorm_list_.append(XNorm_list[i])
      XLast_list_.append(XLast_list[i])
      YNorm_list_.append(YNorm_list[i])
      A_list_.append(A_list[i])
      B_list_.append(B_list[i])
      C_list_.append(C_list[i])
      para_array_.append(para_array[i])
      eig_A_array_.append(eig_A_array[i])

  para_array_ = np.asarray(para_array_); eig_A_array_ = np.asarray(eig_A_array_)
  return para_array_, UNorm_list_, XNorm_list_, XLast_list_, YNorm_list_, A_list_, B_list_, C_list_, eig_A_array_

def fun_MOAS_noinf(A, B, C, D, s, g):
  p = len(C)  # dimension of y
  T = np.linalg.solve(np.identity(len(A))-A, B)
  """ Build the S matrix"""
  S = np.zeros((2*p, p))
  for i in range(0,p):
    S[2*i, i] = 1.0
    S[2*i+1, i] = -1.0
  Kx = np.dot(S,C)
  Lim = np.dot(S,(np.dot(C,T) + D))
  Kr = np.dot(S,D)
  """ Build the core of H and h """
  H = np.concatenate((Kx, Kr),axis=1); h = s

  """ Build the add-on blocks of H and h """
  i = 0
  while i < g :
    i = i + 1
    Kx = np.dot(Kx, A)
    Kr = Lim - np.dot(Kx,T)

    NewBlock = np.concatenate((Kx,Kr), axis=1)
    H = np.concatenate((H,NewBlock)); h = np.concatenate((h,s))
    """ To Insert the ConstRedunCheck """

  return H, h

def fun_RG_SISO(v_0, x, r, H, h, p):
  n = len(x) # dimension of x
  x = np.vstack(x) # x is horizontal array, must convert to vertical for matrix operation
  # because v_0 and r are both scalar, so no need to vstack
  Hx = H[:, 0:n]; Hv = H[:, n:]
  alpha = h - np.dot(Hx,x) - np.dot(Hv,v_0) # alpha is the system remaining vector
  beta = np.dot(Hv, (r-v_0)) # beta is the anticipated response vector with r

  kappa = 1
  for k in range(0,len(alpha)):
    if 0 < alpha[k] and alpha[k] < beta[k]:
      kappa = min(kappa, alpha[k]/beta[k])
    else:
      kappa = kappa
  v = np.asarray(v_0 + kappa*(r-v_0)).flatten()

  return v

def fun_2nd_gstep_calc(x, Hm, hm, A_m, B_m, g):
  n = len(x) # dimension of x
  # x = np.vstack(x) # x is horizontal array, must convert to vertical for matrix operation
  # because v_0 and r are both scalar, so no need to vstack
  Hxm = Hm[:, 0:n]; Hvm = Hm[:, n:]

  T = np.linalg.solve(np.identity(n)-A_m, B_m)
  Ag = np.identity(n)
  for k in range(g+1):
      Ag = np.dot(Ag,A_m)

  alpha = hm - np.dot(Hxm, np.dot(Ag, np.vstack(x)))
  beta = np.dot(Hxm, np.dot((np.identity(n)-Ag),T))
  v_st = []; v_bt = []
  for k in range(0,len(alpha)):
    if beta[k]>0:
      v_st.append(alpha[k]/beta[k])
    elif beta[k]<0:
      v_bt.append(alpha[k]/beta[k])

  v_max = np.asarray(min(v_st))
  v_min = np.asarray(max(v_bt))
  return v_max, v_min