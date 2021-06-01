#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:37:06 2021

@author: samuelkerber
"""

# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#For each output, make sure that the answer is correct, use inputs you know, etc. Do the longer functions
import numpy as np
import pandas as pd
import math
import copy
from termcolor import colored
import cProfile, pstats


pd.options.mode.chained_assignment = None   # default='warn', make sure it is set to 'None' to ensure code interfaces properly with Raven

def wind_cap(WC,normalizedWind):
  gen_wind = WC * normalizedWind            #gen_wind is hourly wind generation, WC is Wind Capacity [MW], normalized wind is hourly production factor of wind [from 0-1]
  cost_wind = WC * 136.0727945*1000         #(136.07 is annualized fixed cost of wind ($/kw-yr), converted to $/MW-yr)
  return gen_wind, cost_wind



    
def solar_cap(SC, normalizedSolar):
  gen_solar = SC * normalizedSolar          #gen_solar follows same path as gen_wind
  cost_solar = SC *83.27*1000         #(120.36 is Annualized fixed cost ($/kw-yr) of solar)
  return gen_solar, cost_solar

    


def net_gen(Demand, gen_solar, gen_wind): #function to calculate "netgen" which is hourly demand [MW] after accounting for VRE generation 
  totalGen = Demand
  netGen = totalGen - gen_wind - gen_solar  #netGen is total demand minus generation from solar and wind
  #netGen[netGen<0] = 0 We will curtail over generation later in the process, because we want to capture over generation in storage system
  return totalGen, netGen




def Storage(netGen,C,HC,HD): #function to determine CAES system size, charging/discharging hours, cost
    #Note that the CAES system functions DAILY
    #Input netGen is the net demand after VREs are accounted for
    #input gen_solar and gen_wind are currently unused, and should be removed
    #input C is the capacity of the ctorage system [MW]
    #input E is the efficiency of the storage system, and is currently unused
    #input HC is the required hours to charge, which will be optimized in raven. It is any number between 0 and 1, representing the distribution of available hours to charge
    #input HD is the required discharging hours (i.e. 4 represents 4 hours of discharge)
    #compressor and turbine sizes are derived from the storage capacity C and the required charging/discharging hours
    Demand2 = pd.DataFrame()
    Demand2["Demand (MW)"] = netGen
    decD = HD - math.floor(HD) #if there is a decimal for discharge hours, this isolates it to allow partial discharge later
    upD = math.ceil(HD) #round up the discharging hours to allow indexing
    #print(Demand2)
    #Cap = C #input C is the capacity of the storage system [MW]
    #Eff = E #input E is the efficiency of the system, it is currently not used
    charging = HC * (24-HD) #the hours charged (HC) can be any amount of time between 0 and 24-HoursOfDischarging (HD)
    hrChrUp = math.ceil(charging) #rounding up the charging hours if they are a decimal, to allow indexing
    compressorSize = (C/charging) #determines compressor size [MW]
    #compressorSize = compressorSize.astype(np.float)
    #print("Compressor size is type ",type(compressorSize))
    turbineSize = (C/HD) #determins turbine size [MW]
    # Demand2=pd.DataFrame()
    # Demand2["Demand (MW)"] = netGen
    f = np.linspace(0,365,num=366) #represents the days of the year
    ###RESET F TO THE LINE ABOVE FOR FULL YEAR, THE VALUE BELOW IS ONLY FOR TESTING
    f = [1]    
    #daySolar=pd.DataFrame()
    #dayWind=pd.DataFrame()
    for j in range(len(f)): #loop through every day of the year
        d = j*24
        dayR = Demand2[d:d+24] #isolates the hours within the day
        ####days = pd.DataFrame(day)
        #print(days)
        
        #dayR = day.reset_index() #resets index and creates column "index" representing the hour of the year
        #print("dayR")
        #print(dayR)
        #dayNew = pd.DataFrame(data=dayR)
        #print("day New")
        #print(dayNew)
        #daySort = sorted(day) #sorts values from low to high, used for charging
        daySort=dayR.sort_values(by=['Demand (MW)'])
        daySort = daySort.reset_index()
        #print("daySort")
        #print(daySort)
        dayMax = dayR.sort_values(by=['Demand (MW)'], ascending=False) #sorts values from high to low, used for discharging
        dayMax = dayMax.reset_index()
        #print("dayMax")
        #print(dayMax)
        Cap2 = C #sets capacity of the storage system
        #b=pd.DataFrame(data=daySort[0:hrChrUp]) #isolates the hours used for charging
        #print("b)")
        #print(b)
        #common=b.merge(dayR) #finds the common values between charging hours and the demand array
        #print("common")
        #print(common)
        #unused = dayR[~dayR['Demand (MW)'].isin(common['Demand (MW)'])] #isolated the values not used to charge
        #print("unused, before index reset")
        #print(unused)
        #unused = unused.reset_index() #resets index, creating column "index" referring to the hour of the year
        #print("unused, after")
        #print(unused)

        #####
        ####w = dayR[dayR['Demand (MW)']==dayMax.at[upD,'Demand (MW)']]['Demand (MW)'] - (turbineSize *decD)
        ####Demand2.update(w)
        #####
        for i in range(hrChrUp+1): #loop through the charging periods
          if Cap2<=0: #if we have filled up our storage space, we move on to discharging 
            for k in range(math.ceil(HD)):###-1): #loop through discharge periods
              print("k is ",k)
                     #decD = HD - math.floor(HD) #if there is a decimal for discharge hours, this isolates it to allow partial discharge later
                     #upD = math.ceil(HD) #round up the discharging hours to allow indexing
              #for w in range(24): #loop through the day
              z = dayR[dayR['Demand (MW)']==dayMax.at[k,'Demand (MW)']]['Demand (MW)'] - turbineSize
              Demand2.update(z)
          #for j in day.index(): #loop through the whole day
          else:
            t = dayR[dayR['Demand (MW)']==daySort.at[i,'Demand (MW)']]['Demand (MW)'] + compressorSize
             #if dayR.at[j,'Demand (MW)']==daySort.at[i,'Demand (MW)']: #matches the daily demand to the periods of lowest demand to perform charging 
               #print("dayR is ",dayR.loc[j]['Demand (MW)'], " and daySort is ", daySort.loc[i]['Demand (MW)'])
               #print("compressor size type ",type(compressorSize))
            Cap2 = Cap2-compressorSize #Cap2 represents available storage space, as we are putting energy to the storage, we must reduce the available space
            print("Cap2 is ", Cap2)   
               #dayNew.loc[j]['Demand (MW)'] = dayR.loc[j]['Demand (MW)'] + compressorSize.item() #increase demand to account for energy put into storage
          #value = dayR.loc[j]['Demand (MW)'] + compressorSize#.item()
               #print(value)
          #dayNew.loc[j,'Demand (MW)'] = value
            Demand2.update(t)
               #print(dayNew.loc[j,'Demand (MW)'])
          
                #if dayR.loc[w]['Demand (MW)'] == dayMax.loc[k]['Demand (MW)']: #matches periods of highest demand to discharge on
                #           #print("dayR is ",dayR.loc[w]['Demand (MW)'], " and dayMax is ", dayMax.loc[k]['Demand (MW)'])
                #  val2= dayR.loc[w]['Demand (MW)'] - turbineSize #reduces demand at high points
                #  dayNew.loc[w,'Demand (MW)'] =val2
                            #print(dayNew.loc[w,'Demand (MW)'])
              #w = dayR[dayR['Demand (MW)']==dayMax.at[upD,'Demand (Mw)']]['Demand (MW)'] -= turbineSize *decD
              #Demand2.update(w)
                #if dayR.loc[w]['Demand (MW)'] ==dayMax.loc[upD]['Demand (MW)']: #if there is a decimal value associated with discharging, this performs partial discharge
                    #val3 = dayR.loc[w]['Demand (MW)'] - turbineSize * decD
                    #dayNew.loc[w,'Demand (MW)']=val3
            #Cap2=Cap #once storage is fully discharged, we reset our storage capacity for the next day
            #break
        #if Cap2>0:  #Need to isolate not the over demand periods
        #   hrsCharge = Cap2/compressorSize
        #   hrsChargeint = math.ceil(hrsCharge)
        #   hrsChargedec = hrsCharge - math.floor(hrsCharge)   
        #   for k in range(hrsChargeint):
        #    
        #    #upC = math.ceil(charging)
        #    #print(k)
        #    # print(daySort[k])
        #    #print(k)
        #    for i in range(24):
        #        #print(dayR)
        #        if dayR.loc[i][0] == unused.loc[k][0]:
        #            # print("matched")
        #            # print(dayR.loc[i]["Demand (MW)"], " to ",daySort[k])
        #            dayR.loc[i][0] = dayR.loc[i][0] + compressorSize
        #        if dayR.loc[i][0] == daySort[hrsChargeint]:
        #            dayR.loc[i][0] = dayR.loc[i][0] + compressorSize * hrsChargedec
        #for k in range(math.ceil(HD)):
        #    decD = HD - math.floor(HD)
        #    upD = math.ceil(HD)
        #    for i in range(24):
        #        if dayR.loc[i][0] == dayMax[k]:
        #            # print("matched")
        #            # print(dayR.loc[i]["Demand (MW)"], " to ",dayMax[k])
        #            dayR.loc[i][0] = dayR.loc[i][0] - turbineSize
        #        if dayR.loc[i][0] ==dayMax[upD]:
        #            dayR.loc[i][0] = dayR.loc[i][0] - turbineSize * decD
        # print(dayR)
        #print("dayR should now be ",dayR)
        #dayR.columns = ['index', 'Demand'] #rename dayR columns to be more usable
        #print(dayR)
        #Demand2['Demand (MW)'][dayNew['index']] = dayNew['Demand (MW)'] #change demand values to reflect the new demand after storage use
        #print("Demand2")
        #print(Demand2)
    Demand2[Demand2<0]=0 #curtail any values that are negative
    storageCost = 105.6107158535297*1000*C + 3.00*(charging*compressorSize + HD*turbineSize)*len(f)   #105 is ~annualized fixed cost, 3.00 is VOM, the sum of the hours is daily hourly use. Comp*Eff is turbine size
    # netGen = Demand2
    netGen = Demand2["Demand (MW)"].squeeze() #let netGen now reflect the new demand
    #capacityStorage = Cap
    return  netGen, storageCost, compressorSize, turbineSize#, capacityStorage  #remove capacityStorage, as it is an input (C)


      



def AFC(Cap,Op,Life): #determines the anualized fixed cost (AFC)
    Nom_WACC =   0.076                          # The nominal after-tax weighted average cost of capital (WACC), WACC is calculated as: WACC = [(cost of equity in %) * (% equity in project)] + [(cost of debt in %) * (% debt in project) * (1 – tax rate)]
    infl = 0.02                             # Long Term inflation Rate (~2%)
    CapEx =  Cap                               # Total capital expenditures ($/kW)
    OpEx =   Op                           # Levelized total operating expenditures ($/kW-yr)
    Real_WACC = (1+Nom_WACC)/(1+infl) -1    # The nominal WACC is converted to a real WACC for purposes of computing the real LCOE
    PDL =  Life                              # Project Design Lifetime
    PDLspace = np.linspace(1,20,num=20)   # Used to calculate PVD
    PVD = sum((0.05/(1+Nom_WACC)**PDLspace))  # Present Value of Depreciation
    TR = 0.25                                   # Tax Rate
    TaxAdj = (1 -TR*PVD)/(1-TR)             # Tax Adjustment
    CRF = (Real_WACC *(1+Real_WACC)**PDL)/((1+Real_WACC)**PDL -1)     # Capital Recovery Factor

    LCOE = (CapEx * CRF * TaxAdj + OpEx) # LCOE, ($/MW-year), UNITS???? WHY NOT $/KW-Year???, really levelized cost of fixed Cap costs 
  # Annualized Fixed Costs == LCOE
  # HTSE Annualized Fixed Cost = (Price of selling 24/7 over lifetime (30 yrs)) - Capital Cost
    #print(LCOE)
    return LCOE

def FuelCell(netGen,rSOFC,hstorage,FChrDischarge,FChrCharge,thresholdFC): #Determines the nuclear-rSOFC-H2storage-SOFC loop
  #This system functions DAILY
  Demand = pd.DataFrame()
  Demand["Demand (MW)"] = netGen
  nucCap = hstorage/(FChrCharge*rSOFC["rProd"]) #determines the nuclear capacity based on the amount of H2 storage the production of the rSOFC
  f = np.linspace(0,365,num=366) #create array to loop through days of the year
  ##### restore f (the line above) before running full code!
  #f=[1]
  FCcap = hstorage/FChrDischarge #determines SOFC capacity
  FChrCharge=FChrCharge*(24-FChrDischarge)
  count = 0 #used to determine the amount of hours NG will be used in the system
  for j in range(len(f)): #loop through the days of the year
      d = j*24
      day = netGen[d:d+24] #isolate hours within the day
      days = pd.DataFrame(day)
      dayR = days.reset_index()     
      dayR.columns=["index","Demand (MW)"]
      daySort = pd.DataFrame(data=dayR) #sort values from low to high, used for charging
      daySort.sort_values(by=['Demand (MW)'],inplace=True)
      daySort.reset_index(inplace=True)
      dayMax=pd.DataFrame(data=dayR)
      dayMax.sort_values(by=['Demand (MW)'],ascending=False,inplace=True) #sort values from high to low, used for discharging
      dayMax.reset_index(inplace=True) 
      hcap = hstorage #determines hydrogen storage capacity
      hrDisDec = FChrDischarge - math.floor(FChrDischarge) #determines a decimal associated with the discharging hours, if needed
      hrDisUp = math.ceil(FChrDischarge) #rounds up the discharge hours to allow indexing
      hrChrDec = FChrCharge - math.floor(FChrCharge) #Determines the decimal associated with charging hours, if needed
      hrChrUp = math.ceil(FChrCharge) #rounds up the charging hours to allow indexing
      b=pd.DataFrame()
      b = dayMax[0:hrDisUp] #isolates discharging hours
      c=pd.DataFrame()
      c = daySort[0:hrChrUp] #isolates charging hours
      d = b.append(c)
      common = d.merge(dayR["Demand (MW)"])
      unused = dayR[~dayR["Demand (MW)"].isin(common["Demand (MW)"])] #determines hours when charging and discharging is not performed
      unused = unused.reset_index()
      for i in range(math.ceil(FChrDischarge)): #loop through discharging hours
          for k in range(len(daySort)): #loop through hours in the day
              if dayR.loc[k]["index"]==dayMax["index"][i]: #match highest demand periods
                dayR.loc[k]["Demand (MW)"]= dayR.loc[k]["Demand (MW)"] - FCcap - nucCap #reduce demand by nuclear capacity and SOFC capacity
              if dayR.loc[k]["index"] == dayMax["index"][hrDisUp]: #perform partial discharge on discharge hours where decimal exists
                 dayR.loc[k]["Demand (MW)"] = dayR.loc[k]["Demand (MW)"]- FCcap *hrDisDec - nucCap *hrDisDec
      for i in range(len(unused)): #loop through unused hours
          for k in range(len(daySort)): #loop through hours in the day
              if dayR.loc[k]["index"] == unused.loc[i]["index"]: #matches unused hours
                 dayR.loc[k]["Demand (MW)"] = dayR.loc[k]["Demand (MW)"] - nucCap #reduces demand at unused hours by nuclear capacity
                 if dayR.loc[k]["Demand (MW)"] > thresholdFC: #determines a threshold value, above which NG will fuel the SOFC
                    dayR.loc[k]["Demand (MW)"] = dayR.loc[k]["Demand (MW)"] - FCcap #reduces demand at unused hours by SOFC capacity due to fueling from NG
                    count = count +1  #counts the hours NG is used       
      for i in range(len(dayR)): #replace old demands with new demands reflecting modifications from the SOFC system
          Demand["Demand (MW)"][dayR.loc[i]["index"]] = dayR.loc[i]["Demand (MW)"]           
  capcost = rSOFC["nCapCost"] + rSOFC["rCapCost"] + rSOFC["sCapCost"] #determines total system capital cost
  FOM = rSOFC["nFOM"] + rSOFC["rFOM"] + rSOFC["sVOM"] #determines total system FOM cost
  life = 20 #set lifetime, used for AFC function
  nLCOE = AFC(rSOFC["nCapCost"],rSOFC["nFOM"],rSOFC["nLife"]) #run AFC on nuclear
  rLCOE = AFC(rSOFC["rCapCost"],rSOFC["rFOM"],rSOFC["rLife"]) #run AFC on rSOFC
  sLCOE = AFC(rSOFC["sCapCost"],rSOFC["sFOM"],rSOFC["sLife"]) #run AFC on SOFC
  fixcost = nLCOE * nucCap + rLCOE * hstorage / (FChrCharge*rSOFC["rProd"]) + sLCOE * hstorage/(FChrDischarge * rSOFC["sUtility"]) + rSOFC["hStore"]*hstorage #determine annualized fixed cost of system 
  varcost = rSOFC["nVOM"] * (8760) + rSOFC["rVOM"] * FChrCharge + rSOFC["sVOM"]*FChrDischarge #determine VOM of system
  NGcost = rSOFC["sVOM"] * count * rSOFC["NGcost"] * rSOFC["NGeff"] #determine cost associated with the Natural Gas use
  return nucCap,FCcap,capcost,FOM,nLCOE,rLCOE,sLCOE,fixcost,varcost,NGcost,count,Demand




def Syngas(netGen,Syn,synstorage,SynhrDischarge,SynhrCharge,thresholdSyn): #Determines the nuclear-rSOFC-H2storage-SOFC loop
  #This system functions DAILY
  Demand = pd.DataFrame()
  Demand["Demand (MW)"] = netGen
  nucCap = synstorage/(SynhrCharge*Syn["rProd"]) #determines the nuclear capacity based on the amount of H2 storage the production of the rSOFC
  f = np.linspace(0,365,num=366) #create array to loop through days of the year
  ##### restore f (the line above) before running full code!
  #f=[1]
  FCcap = synstorage/SynhrDischarge #determines SOFC capacity
  coalCap = synstorage/(SynhrCharge*Syn["cCarb"])
  SynhrCharge=SynhrCharge*(24-SynhrDischarge)
  count = 0 #used to determine the amount of hours NG will be used in the system
  for j in range(len(f)): #loop through the days of the year
      d = j*24
      day = netGen[d:d+24] #isolate hours within the day
      days = pd.DataFrame(day)
      dayR = days.reset_index()     
      dayR.columns=["index","Demand (MW)"]
      dayR["Demand (MW)"] = dayR["Demand (MW)"] - coalCap
      daySort = pd.DataFrame(data=dayR) #sort values from low to high, used for charging
      daySort.sort_values(by=['Demand (MW)'],inplace=True)
      daySort.reset_index(inplace=True)
      dayMax=pd.DataFrame(data=dayR)
      dayMax.sort_values(by=['Demand (MW)'],ascending=False,inplace=True) #sort values from high to low, used for discharging
      dayMax.reset_index(inplace=True) 
      syncap = synstorage #determines hydrogen storage capacity
      hrDisDec = SynhrDischarge - math.floor(SynhrDischarge) #determines a decimal associated with the discharging hours, if needed
      hrDisUp = math.ceil(SynhrDischarge) #rounds up the discharge hours to allow indexing
      hrChrDec = SynhrCharge - math.floor(SynhrCharge) #Determines the decimal associated with charging hours, if needed
      hrChrUp = math.ceil(SynhrCharge) #rounds up the charging hours to allow indexing
      b=pd.DataFrame()
      b = dayMax[0:hrDisUp] #isolates discharging hours
      c=pd.DataFrame()
      c = daySort[0:hrChrUp] #isolates charging hours
      d = b.append(c)
      common = d.merge(dayR["Demand (MW)"])
      unused = dayR[~dayR["Demand (MW)"].isin(common["Demand (MW)"])] #determines hours when charging and discharging is not performed
      unused = unused.reset_index()
      for i in range(math.ceil(SynhrDischarge)): #loop through discharging hours
          for k in range(len(daySort)): #loop through hours in the day
              if dayR.loc[k]["index"]==dayMax["index"][i]: #match highest demand periods
                dayR.loc[k]["Demand (MW)"]= dayR.loc[k]["Demand (MW)"] - FCcap - nucCap # -coalCap #reduce demand by nuclear capacity and SOFC capacity
              if dayR.loc[k]["index"] == dayMax["index"][hrDisUp]: #perform partial discharge on discharge hours where decimal exists
                 dayR.loc[k]["Demand (MW)"] = dayR.loc[k]["Demand (MW)"]- FCcap *hrDisDec - nucCap *hrDisDec #-coalCap*hrDisDec
      for i in range(len(unused)): #loop through unused hours
          for k in range(len(daySort)): #loop through hours in the day
              if dayR.loc[k]["index"] == unused.loc[i]["index"]: #matches unused hours
                 dayR.loc[k]["Demand (MW)"] = dayR.loc[k]["Demand (MW)"] - nucCap #-coalCap #reduces demand at unused hours by nuclear capacity
                 if dayR.loc[k]["Demand (MW)"] > thresholdSyn: #determines a threshold value, above which NG will fuel the SOFC
                    dayR.loc[k]["Demand (MW)"] = dayR.loc[k]["Demand (MW)"] - FCcap #reduces demand at unused hours by SOFC capacity due to fueling from NG
                    count = count +1  #counts the hours NG is used       
      for i in range(len(dayR)): #replace old demands with new demands reflecting modifications from the SOFC system
          Demand["Demand (MW)"][dayR.loc[i]["index"]] = dayR.loc[i]["Demand (MW)"]           
  capcost = Syn["nCapCost"] + Syn["rCapCost"] + Syn["sCapCost"] +Syn["cCap"] #determines total system capital cost
  FOM = Syn["nFOM"] + Syn["rFOM"] + Syn["sVOM"] +Syn["cCap"] #determines total system FOM cost
  life = 20 #set lifetime, used for AFC function
  nLCOE = AFC(Syn["nCapCost"],Syn["nFOM"],Syn["nLife"]) #run AFC on nuclear
  rLCOE = AFC(Syn["rCapCost"],Syn["rFOM"],Syn["rLife"]) #run AFC on rSOFC
  sLCOE = AFC(Syn["sCapCost"],Syn["sFOM"],Syn["sLife"]) #run AFC on SOFC
  cLCOE = AFC(Syn["cCap"],Syn["cFOM"],Syn["cLife"])
  fixcost = nLCOE * nucCap + rLCOE * synstorage / (SynhrCharge*Syn["cCarb"]) + sLCOE * synstorage/(SynhrDischarge * Syn["sUtility"]) + Syn["hStore"]*synstorage +cLCOE*coalCap #determine annualized fixed cost of system 
  varcost = Syn["nVOM"] * (8760) + Syn["rVOM"] * SynhrCharge + Syn["sVOM"]*SynhrDischarge +Syn["cVOM"]*8760 #determine VOM of system
  NGcost = Syn["sVOM"] * count * Syn["NGcost"] * Syn["NGeff"] #determine cost associated with the Natural Gas use
  return nucCap,coalCap,FCcap,capcost,FOM,nLCOE,rLCOE,sLCOE,fixcost,varcost,NGcost,count,Demand





def _readMoreXML(raven,xmlNode): #read values in from raven
  raven.demandDataFile = None
  raven.solarID   = None
  raven.windID   = None
  raven.demandID = None
  for child in xmlNode:
    if child.tag == 'demandDataFile':  #set demand file in Raven
      raven.demandDataFile = child.text
    if child.tag == 'solarID':          #set in Raven
      raven.solarID = child.text
    if child.tag == 'windID':           #set in Raven
      raven.windID = child.text
    if child.tag == 'demandID':         #set in Raven
      raven.demandID = child.text
  if raven.demandDataFile is None:
    raise IOError("demand file not specified!")
  if raven.solarID is None:
    raise IOError("solarID not specified!")
  if raven.windID is None:
    raise IOError("windID not specified!")
  if raven.demandID is None:
    raise IOError("demandID not specified!")

def initialize(raven,runInfoDict,inputFiles): #set values from raven
  Generation = pd.read_excel(raven.demandDataFile)    #Read demand file
  raven.solar = Generation[raven.solarID]             #Read solar generation column
  raven.wind = Generation[raven.windID]               #Read wind generation column
  maxSolar = max(raven.solar)                         #Find maximum solar generation, this will be our solar capacity production factor of 1
  maxWind = max(raven.wind)                           #Find maximum wind generation, this will be our wind capacity production factor of 1
  raven.demand = Generation[raven.demandID]           #Set hourly demand profile
  raven.normalizedSolar = raven.solar/maxSolar        #Normalize solar production between 0 and 1, with one being the index of maxSolar
  raven.normalizedWind = raven.wind/maxWind           #See normalizedSolar
  raven.normalizedWind = raven.normalizedWind.to_numpy().flatten()    #Flatten for output
  raven.normalizedSolar = raven.normalizedSolar.to_numpy().flatten()  #Flatten for output
  raven.demand = raven.demand.to_numpy().flatten()                    #Flatten for output
  print(maxSolar)
  print(maxWind)
  return

def evaluate_cost(LL,self,taxRate): #Determines cost curve at each MW of generation
        self["cost"] = self["FOM"] + ((LL['counts']-1)*(self["VOM"]  + (self["CarbonEmission"] *taxRate)))
        self["Carbon"] = self["CarbonEmission"]*(LL['counts']-1)
        #Carbon Emission should be ton/MWh
        

def load_levels(netGen,resolution, maxLoad): #Determines number of steps to use for given resolution, 1 MW should ALWAYS be used. If this is changed, scale all costs accordingly
    #steps = maxLoad / resolution
    #steps = math.ceil(steps)
    #print(maxLoad)
    loadCurve = netGen
    loadLevel = np.linspace(0,math.ceil(maxLoad), num=math.ceil(maxLoad)+1) #loadLevel is set from 0-maximum load, with step size equivalent to 1 MW
    ### To do: 
    ### 
    return loadCurve, loadLevel

def counts(loadLevel, loadCurve): #Counts number of operational hours at each load level
    
    #countArray=np.zeros(len(loadLevel))
    #for idx, i in enumerate(loadLevel): #loop through load levels
    #    counts = np.count_nonzero(loadCurve>i) #count number of instances above a given load level
    #    countArray[idx] = counts+1
    #columns = ['LoadLevel','counts']
    #temp = zip(loadLevel,countArray)
    #LL = pd.DataFrame(temp,columns=columns)
    #LL.columns = ["LoadLevel"]
    #LL['counts'] = countArray
    ###y = np.bincount(loadCurve) 
    ###z = np.cumsum(y)
    ###
    #z = np.insert(z,-1,z[-1]) 
    ###z = len(loadCurve) - z +1
    y = np.bincount(loadCurve) 
    z = np.cumsum(y)
    ###z = np.insert(z,-1,z[-1]) 
    z = len(loadCurve) - z +1
    print("Length of z is ", len(z))
    print("Length of loadLevel is ", len(loadLevel))
    LL = pd.DataFrame()
    LL['LoadLevel'] = loadLevel
    LL['counts'] = z
    return LL
    
    



        

def find_caps(LL,*kwargs): #Finds the optimal capacity of generators 
    min_cost_array = LL['LoadLevel']
    cost_array = pd.DataFrame()
    cost_array['LoadLevel'] = min_cost_array
    df = pd.DataFrame()
    for i in range(len(kwargs)): #fill columns (named after the generator) with the associated cost at each load level
        # print(kwargs[i])
        cost_array[kwargs[i]["Name"]] = kwargs[i]["cost"]
    #d = 'Minus'
    #for i in range(len(kwargs)): #determines difference in cost at a load level between generators
    #    for j in range(len(kwargs)):
    #        count = 0
    #        
    #        if i == j:
    #            continue
    #        else:
    #            b = kwargs[i]["cost"]-kwargs[j]["cost"]
    #            c = kwargs[i]["Name"] + d + kwargs[j]["Name"]
    #            count = count+1
    #            
    #            df.loc[:,c] = b           
    #fullCostArray = pd.concat([cost_array,df], axis=1)
    #absCostArray =  abs(df)
    #minVals = absCostArray.min(axis=0)
    #minIdx = absCostArray.idxmin(axis=0)
    #mins = pd.concat([minIdx,minVals], axis=1)    
    mincost = cost_array.iloc[:,1:]
    mincostline = mincost.min(axis=1) #determines the minimum cost at any load level
    capacities=pd.DataFrame()
    for i in range(len(kwargs)):
        try:
            intersect = np.where(mincostline == kwargs[i]["cost"]) #determines intersection points
            minIntersect = np.min(intersect) #determines lowest intersection for generator
            maxIntersect = np.max(intersect) #determines maximum intersection for generator
            capacity = maxIntersect - minIntersect #capacity is the difference between the intersections
            capacities.loc[1,kwargs[i]["Name"]] = minIntersect #value when generator turns on
            capacities.loc[2,kwargs[i]["Name"]] = maxIntersect #value when generator is oeprating at full capacity, THIS VALUE IS INCLUSIVE
            capacities.loc[3,kwargs[i]["Name"]] = capacity
            print("Generator included is",kwargs[i]["Name"]) 
            LL.loc[minIntersect:maxIntersect,"counts"] = LL.loc[minIntersect:maxIntersect,"counts"] -1
            capacities.loc[4,kwargs[i]["Name"]] = LL.loc[minIntersect:maxIntersect,"counts"].sum() #determines total generation form generator
            print("MW generated from ", kwargs[i]["Name"],"is ",LL.loc[minIntersect:maxIntersect,"counts"].sum())
            carb = kwargs[i]["CarbonEmission"] * LL.loc[minIntersect:maxIntersect,"counts"]
            capacities.loc[5,kwargs[i]["Name"]] = carb.sum() #determines total carbon emission over the year
        except:
                print("No intersections")   
    for i in range(len(kwargs)):
      if not kwargs[i]["Name"] in capacities: #adds all generators not included with values of zero for all fields
        capacities[kwargs[i]["Name"]]=[0,0,0,0,0]
        
    print("The Following Tables has row one as the minimum load for which the generator will turn on")
    print("The second row is the level where the generator is running at full installed capacity, INCLUDING this level")
    print("The third row is the total capacity of the generator")    
    print("The fourth row is the total amount the unit generated over the year")
    print("The fifth row is the total tons of carbon emitted by the source over the year")
    print(capacities)
    
    totalCost = np.sum(mincostline)
    print("")
    print ("The traditional generators cost is", totalCost)
    return mincostline, capacities, totalCost



def plots(loadLevel,*kwargs):
    for i in range(len(kwargs)):
        name = kwargs[i]['Name']
    #     plt.plot(loadLevel,kwargs[i]["cost"],label=name)
    # plt.legend(loc='upper left')
    # plt.xlabel("Load Level (MW)")
    # plt.ylabel("Total Cost ($/MW-year)")
    
def run2(netGen, TR, *kwargs):
    
    loadCurve = netGen 
    print("loadCurve established")
    load = loadCurve
    maxLoad = max(load)
    taxRate = TR                                                    #$/ton CO2
    loadCurve, loadLevel = load_levels(netGen,1,maxLoad)
    print("load_levels ran")
    LL = counts(loadLevel,loadCurve)
    print("counts ran")
    for i in range(len(kwargs)):
        evaluate_cost(LL, kwargs[i],taxRate)
    print("evaluate_cost ran")
    mincostline, capacities, totalCost = find_caps(LL,*kwargs)
    print("find caps worked")
    # plot = plots(loadLevel,*kwargs)
    # plotmin = plt.plot(loadLevel, mincostline, label="Minimum Cost")
    # plt.legend(loc='upper left')
    # plt.xlabel("Load Level (MW)")
    # plt.ylabel("Total Cost ($/MW-year)")
    return capacities, totalCost, mincostline
   

# Nuclear={"Name":"Nuclear","FOM":6317,"VOM":2.56,"CarbonEmission":0}
# Coal={"Name":"Coal","FOM":4652,"VOM":7.06,"CarbonEmission":0}
# FC={"Name":"Fuel Cell","FOM":7339,"VOM":0.59,"CarbonEmission":0}        




# $2/kg H2
Coal = {"Name":"Coal","FOM":307059,"VOM":30.92976827,"CarbonEmission":0.997903}
Coal30 = {"Name":"Coal30","FOM":384716.945,"VOM":36.91522463,"CarbonEmission":0.6985321}
Coal90 = {"Name":"Coal90","FOM":485500.941,"VOM":49.24778325,"CarbonEmission":0.0997903}
# FC = {"Name":"Fuel Cell","FOM":683245.1647,"VOM":0.59,"CarbonEmission":0}
Nuclear = {"Name":"Nuclear","FOM":405690.9528,"VOM":0,"CarbonEmission":0}
NUCHTSE = {"Name":"NucHTSE","FOM":32711.20087,"VOM":53.47593583,"CarbonEmission":0}
CC221 = {"Name":"CC221","FOM":81647.00161,"VOM":24.165,"CarbonEmission":0.408233}
CC111 ={"Name":"CC111","FOM":108131.8966,"VOM":25.0585,"CarbonEmission":0.408233}
CC111_90 = {"Name":"CC111_90","FOM":207451.786,"VOM":30.774,"CarbonEmission":0.0408233}
IC = {"Name":"IC","FOM":180053.3027,"VOM":34.7225,"CarbonEmission":0.408233}
CTLM = {"Name":"CTLM","FOM":101477.6899,"VOM":36.634,"CarbonEmission":0.408233}
CTGE = {"Name":"CTGE","FOM":58686.54713,"VOM":39.1675,"CarbonEmission":0.408233}
rSOFC = {"Name":"rSOFC","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.
Syn = {"Name":"Syn","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95,"cCap":5876,"cFOM":59000,"cVOM":11,"cCarb":.99, "cLife":40} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.


# $1.5/kg H2
#Coal = {"Name":"Coal","FOM":307059,"VOM":30.92976827,"CarbonEmission":0.997903}
#Coal30 = {"Name":"Coal30","FOM":384716.945,"VOM":36.91522463,"CarbonEmission":0.6985321}
#Coal90 = {"Name":"Coal90","FOM":485500.941,"VOM":49.24778325,"CarbonEmission":0.0997903}
# FC = {"Name":"Fuel Cell","FOM":683245.1647,"VOM":0.59,"CarbonEmission":0}
#Nuclear = {"Name":"Nuclear","FOM":405690.9528,"VOM":0,"CarbonEmission":0}
#NUCHTSE = {"Name":"NucHTSE","FOM":149823.5003,"VOM":31.87255187,"CarbonEmission":0}
#CC221 = {"Name":"CC221","FOM":81647.00161,"VOM":24.165,"CarbonEmission":0.408233}
#CC111 ={"Name":"CC111","FOM":108131.8966,"VOM":25.0585,"CarbonEmission":0.408233}
#CC111_90 = {"Name":"CC111_90","FOM":207451.786,"VOM":30.774,"CarbonEmission":0.0408233}
#IC = {"Name":"IC","FOM":180053.3027,"VOM":34.7225,"CarbonEmission":0.408233}
#CTLM = {"Name":"CTLM","FOM":101477.6899,"VOM":36.634,"CarbonEmission":0.408233}
#CTGE = {"Name":"CTGE","FOM":58686.54713,"VOM":39.1675,"CarbonEmission":0.408233}
#rSOFC = {"Name":"rSOFC","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.
#Syn = {"Name":"Syn","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95,"cCap":5876,"cFOM":59000,"cVOM":11,"cCarb":.99, "cLife":40} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.

# $1.25/kg H2
#Coal = {"Name":"Coal","FOM":307059,"VOM":30.92976827,"CarbonEmission":0.997903}
#Coal30 = {"Name":"Coal30","FOM":384716.945,"VOM":36.91522463,"CarbonEmission":0.6985321}
#Coal90 = {"Name":"Coal90","FOM":485500.941,"VOM":49.24778325,"CarbonEmission":0.0997903}
# FC = {"Name":"Fuel Cell","FOM":683245.1647,"VOM":0.59,"CarbonEmission":0}
#Nuclear = {"Name":"Nuclear","FOM":405690.9528,"VOM":0,"CarbonEmission":0}
#NUCHTSE = {"Name":"NucHTSE","FOM":208379.6501,"VOM":25.18805989,"CarbonEmission":0}
#CC221 = {"Name":"CC221","FOM":81647.00161,"VOM":24.165,"CarbonEmission":0.408233}
#CC111 ={"Name":"CC111","FOM":108131.8966,"VOM":25.0585,"CarbonEmission":0.408233}
#CC111_90 = {"Name":"CC111_90","FOM":207451.786,"VOM":30.774,"CarbonEmission":0.0408233}
#IC = {"Name":"IC","FOM":180053.3027,"VOM":34.7225,"CarbonEmission":0.408233}
#CTLM = {"Name":"CTLM","FOM":101477.6899,"VOM":36.634,"CarbonEmission":0.408233}
#CTGE = {"Name":"CTGE","FOM":58686.54713,"VOM":39.1675,"CarbonEmission":0.408233}
#rSOFC = {"Name":"rSOFC","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.
#Syn = {"Name":"Syn","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95,"cCap":5876,"cFOM":59000,"cVOM":11,"cCarb":.99, "cLife":40} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.

# $1/kg H2
#Coal = {"Name":"Coal","FOM":307059,"VOM":30.92976827,"CarbonEmission":0.997903}
#Coal30 = {"Name":"Coal30","FOM":384716.945,"VOM":36.91522463,"CarbonEmission":0.6985321}
#Coal90 = {"Name":"Coal90","FOM":485500.941,"VOM":49.24778325,"CarbonEmission":0.0997903}
# FC = {"Name":"Fuel Cell","FOM":683245.1647,"VOM":0.59,"CarbonEmission":0}
#Nuclear = {"Name":"Nuclear","FOM":405690.9528,"VOM":0,"CarbonEmission":0}
#NUCHTSE = {"Name":"NucHTSE","FOM":266935.7998,"VOM":18.50356791,"CarbonEmission":0}
#CC221 = {"Name":"CC221","FOM":81647.00161,"VOM":24.165,"CarbonEmission":0.408233}
#CC111 ={"Name":"CC111","FOM":108131.8966,"VOM":25.0585,"CarbonEmission":0.408233}
#CC111_90 = {"Name":"CC111_90","FOM":207451.786,"VOM":30.774,"CarbonEmission":0.0408233}
#IC = {"Name":"IC","FOM":180053.3027,"VOM":34.7225,"CarbonEmission":0.408233}
#CTLM = {"Name":"CTLM","FOM":101477.6899,"VOM":36.634,"CarbonEmission":0.408233}
#CTGE = {"Name":"CTGE","FOM":58686.54713,"VOM":39.1675,"CarbonEmission":0.408233}
#rSOFC = {"Name":"rSOFC","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.
#Syn = {"Name":"Syn","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":25,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95,"cCap":5876,"cFOM":59000,"cVOM":11,"cCarb":.99, "cLife":40} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.

#Coal = {"Name":"Coal","FOM":307059,"VOM":30.92976827,"CarbonEmission":0}
#Coal30 = {"Name":"Coal30","FOM":384716.945,"VOM":36.91522463,"CarbonEmission":0}
#Coal90 = {"Name":"Coal90","FOM":485500.941,"VOM":49.24778325,"CarbonEmission":0}
# FC = {"Name":"Fuel Cell","FOM":683245.1647,"VOM":0.59,"CarbonEmission":0}
#Nuclear = {"Name":"Nuclear","FOM":405690.9528,"VOM":0,"CarbonEmission":0}
#NUCHTSE = {"Name":"NucHTSE","FOM":32711.20087,"VOM":53.47593583,"CarbonEmission":0}
#CC221 = {"Name":"CC221","FOM":81647.00161,"VOM":24.165,"CarbonEmission":0}
#CC111 ={"Name":"CC111","FOM":108131.8966,"VOM":25.0585,"CarbonEmission":0}
#CC111_90 = {"Name":"CC111_90","FOM":207451.786,"VOM":30.774,"CarbonEmission":0}
#IC = {"Name":"IC","FOM":180053.3027,"VOM":34.7225,"CarbonEmission":0}
#CTLM = {"Name":"CTLM","FOM":101477.6899,"VOM":36.634,"CarbonEmission":0}
#CTGE = {"Name":"CTGE","FOM":58686.54713,"VOM":39.1675,"CarbonEmission":0}

def run(raven,Input):
  profiler = cProfile.Profile()
  profiler.enable()
  raven.TR
  raven.HC
  raven.HD
  gen_solar, cost_solar = solar_cap(raven.SC,raven.normalizedSolar)
  print("solar_cap ran")
  gen_wind, cost_wind = wind_cap(raven.WC,raven.normalizedWind)
  print("wind_cap ran")
  
  totalGen, netGen = net_gen(raven.demand, gen_solar, gen_wind)
  print("net_gen ran")
  #print(len(netGen))
####  
  netGen, storageCost, compressorSize, turbineSize = Storage(netGen,raven.C,raven.HC,raven.HD)
  print("Storage ran")
  VREcost = cost_solar + cost_wind
  print("VREcost established")
  
  #nucCap,FCcap,capcost,FOM,nLCOE,rLCOE,sLCOE,fixcost,varcost,NGcost,count,Demand = FuelCell(netGen,rSOFC,raven.hstorage,raven.FChrDischarge,raven.FChrCharge,raven.thresholdFC)
  #print("Fuel Cell ran")
  #H2FCcost = fixcost+varcost+NGcost
  #nucCapSyn,coalCapSyn,FCcapSyn,capcostSyn,FOMSyn,nLCOESyn,rLCOESyn,sLCOESyn,fixcostSyn,varcostSyn,NGcostSyn,countSyn,DemandSyn = Syngas(netGen,Syn,raven.synstorage,raven.SynhrDischarge,raven.SynhrCharge,raven.thresholdSyn)
  #print("Syngas ran")
  #Syncost = fixcostSyn+varcostSyn+NGcostSyn
  #conversion pandas to numpy arrays
  raven.time = np.asarray(list(range(len(totalGen))))
  raven.totalGen = totalGen.flatten()
#######
  raven.netGen = netGen.to_numpy()
  #raven.netGen = netGen.flatten()
  raven.gen_wind = gen_wind.flatten()
  raven.gen_solar = gen_solar.flatten()
  arg = copy.deepcopy([Coal,Coal30,Coal90,NUCHTSE,Nuclear,CC221,CC111,CC111_90,IC,CTLM,CTGE])
  capacities, totalCost, mincostline = run2(netGen,raven.TR,*arg)
  print("run2 worked")
  for generatorID in capacities:
    minL,maxL,c,generated,carbon = capacities[generatorID]
    minLoad = np.zeros(len(raven.time))
    minLoad[:] = minL
    maxLoad = np.zeros(len(raven.time))
    maxLoad[:] = maxL
    cap = np.zeros(len(raven.time))
    cap[:] = c
    Generated = np.zeros(len(raven.time))
    Generated[:] = generated
    TotalCarbon = np.zeros(len(raven.time))
    TotalCarbon[:] = carbon
    raven.__dict__[generatorID+"_minLoad"]  = minLoad
    raven.__dict__[generatorID+"_maxLoad"]  = maxLoad
    raven.__dict__[generatorID+"_capacity"] = cap
    raven.__dict__[generatorID+"_totalGenerated"] = Generated
    raven.__dict__[generatorID+"_totalCarbon"] = TotalCarbon
  
  raven.totalCost = np.zeros(len(raven.time))
  raven.totalCost[:] = totalCost
  
  raven.mincostline = mincostline.to_numpy().flatten()

  # conversion scalar to arrays
  raven.VREcost = np.zeros(len(raven.time))
  raven.VREcost[:] = VREcost
  raven.cost_solar = np.zeros(len(raven.time))
  raven.cost_solar[:] = cost_solar
  raven.cost_wind = np.zeros(len(raven.time))
  raven.cost_wind[:] = cost_wind
  raven.cost_wind = np.zeros(len(raven.time))
  raven.cost_wind[:] = cost_wind
#####
  systemcost = totalCost + VREcost + storageCost #+ H2FCcost + Syncost
  raven.systemcost = np.zeros(len(raven.time))
  raven.systemcost[:] = systemcost
  print(raven.systemcost.shape)
  print(raven.time.shape)
  print("solar is")
  print(raven.SC)
  print("Solar generated ", gen_solar.sum(),"MW over the entire year")
  print("Wind is")
  print(raven.WC)
  print("Wind generated ", gen_wind.sum(), "MW over the entire year")
  print("Capacity of storage is ",raven.C)
  print("Compressor Size is ",compressorSize)
  print("Turbine size is ",turbineSize)
  print("Storage cost is ",storageCost, " For the year")
  print("Total System Cost is")
  print(raven.systemcost[1])
  try:
    TC = capacities.loc[5,:]
    print("Total carbon [tons] generated throughout the year for the entire system is")
    print (TC.sum())
    carbonGenerated = TC.sum()
    raven.carbonGenerated = np.zeros(len(raven.time))
    raven.carbonGenerated[:]=carbonGenerated
  except:
    print("Carbon either not generated or failed")
  profiler.disable()
  stats = pstats.Stats(profiler).sort_stats('tottime')
  #stats.print_stats() 
  filename = 'profile.prof'  # You can change this if needed
  profiler.dump_stats(filename)


  


#if __name__ == '__run__':
    
#    profiler = cProfile.Profile()
#    profiler.enable()
#    run(raven,Input)
#    profiler.disable()
#    stats = pstats.Stats(profiler).sort_stats('tottime')
#    stats.print_stats()  

