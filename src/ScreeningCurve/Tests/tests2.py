#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import math
import copy
from termcolor import colored
import NewSCM as SCM

pd.options.mode.chained_assignment = None

def test_wind_cap(): #function test to see if wind_cap functions correctly
    gen_wind, cost_wind = SCM.wind_cap(1,1)
    test_net = 1
    if test_net - gen_wind ==0:
       print(colored("Wind_cap net calculation Works","green"))
    if test_net - gen_wind != 0:
       print (colored("Wind Cap netgen failed","red"))
    value = 136.0727945*1000
    if cost_wind == value:
       print(colored("Wind cap cost calculation works","green"))
    if cost_wind != value:
       print(colored("wind cap cost calculaiton fails","red"))

def test_solar_cap(): #function test to see if solar_cap functions correctly
    gen_solar, cost_solar = SCM.solar_cap(1,1)
    test_net = 1
    if test_net - gen_solar ==0:
       print(colored("Solar cap net calculation Works","green"))
    if test_net - gen_solar != 0:
       print (colored("Solar Cap netgen failed","red"))
    value = 83.27*1000
    if cost_solar==value:
       print(colored("Solar cap cost calculation works","green"))
    if cost_solar!=value:
       print(colored("Solar cap cost calculaiton fails","red"))


def test_net_gen(): #Test to see that netgen functions correctly
    simDemand = pd.DataFrame([2,2,2]) #simulated demand curve
    simWind = pd.DataFrame([1,1,1]) #simulated wind generation
    simSolar = pd.DataFrame([1,1,1]) #simulated solar generation
    
    totalGen, netGen = SCM.net_gen(simDemand,simSolar,simWind)
    netTest = netGen.sum()
    if netTest[0] ==0:
       print(colored("Net Gen functions correctly for first pass (VREs)","green"))
    if netTest[0] !=0:
       print(colored("Net Gen fails for VREs","red"))


def test_storage(): #test to make sure the storage system works 
    netGen1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    HC = 0.5
    HD = 4
    netGennc, storageCost,compressorSize,turbineSize = SCM.Storage(netGen1,100,HC,HD)
    print(netGennc)
    if compressorSize == 10:
       print(colored("CAES compressor size worked","green"))
    if compressorSize != 10:
       print(colored("CAES compressor size failed","red"))
    if turbineSize == 25:
       print(colored("CAES turbine size worked","green"))
    if turbineSize !=25:
       print(colored("CAES turbine size failed","red"))
    f = np.linspace(0,365,num=366)
    #f=[1]
    cost = 105.6107158535297*1000*100 + 3.00*(10*10 + 4*25)
    if cost == storageCost:
       print(colored("CAES cost works","green"))
    if cost != storageCost:
       print(colored("CAES cost fails","red"))
       print("CAES cost was ", storageCost)
       print("But should have been ",cost)
    testGen = [11,12,13,14,15,16,17,18,19,20,11,12,13,14,15,16,17,18,19,20,0,0,0,0]
    testnetgen = testGen - netGennc    
    testnetgen = testnetgen.sum()
    if testnetgen == 0:
       print(colored("CAES netGen functions correctly with no curtailment","green"))
    if testnetgen != 0:
       print(colored("CAES netGen does not work with no curtailment","red"))
       print("sum of testnetgen was ", testnetgen)
       print("sum of testnetgen should be ",0)
    netgen2 = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    netGenpeak, storageCost, compressorSize,turbineSize = SCM.Storage(netgen2,100,HC,HD)
    testpeak = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,0,0,0] 
    testpeaksum = testpeak-netGenpeak
    testpeaksum = testpeaksum.sum()
    if testpeaksum ==0:
       print(colored("Storage curtailment worked","green"))
    if testpeaksum !=0:
       print(colored("Storage curtailment failed","red"))

def test_FuelCell(): #test that the rSOFC loop functions correctly
  netGen=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
  rSOFC = {"Name":"rSOFC","nCapCost":3000,"nFOM":405000,"nVOM":10,"nLife":40,"rCapCost":600,"rFOM":6,"rVOM":0.5,"rProd":1,"rLife":20,"sCapCost":600,"sFOM":6,"sVOM":0.5,"sLife":20,"sUtility":20,"hStore":450,"NGcost":5,"NGeff":0.95} #rProd is h2 production kg H2/MWh, hStore is hydrogen storage $/kg h2, sUtility is how much h2 used per MW in SOFC, NGcost is cost ($/MWh) of ng,NGeff is efficiency of SOFC when using NG.
  nucCap,FCcap,capcost,FOM,nLCOE,rLCOE,sLCOE,fixcost,varcost,NGcost,count,Demand = SCM.FuelCell(netGen,rSOFC,10,5,5,10)
  if nucCap ==2:
    print(colored("rSOFC nuclear capacity works","green"))
  else:
    print(colored("rSOFC nuclear capacity fails","red"))
  if FCcap == 2:
    print(colored("rSOFC fuel cell capacity works","green"))
  else:
    print(colored("rSOFC fuel cell capacity fails","red"))
  demandCheck =pd.DataFrame()
  demandCheck["Demand (MW)"] = [1,2,3,4,5,4,5,6,7,8,9,10,9,10,11,12,13,14,15,16,17,18,19,20]
  demandsum=demandCheck["Demand (MW)"] - Demand["Demand (MW)"]
  demandsum=demandsum.sum()
  if  demandsum == 0:
    print(colored("rSOFC demand modifications are succesful","green"))
  else:
    print(colored("rSOFC demand modication failed","red"))


def test_SCM(): #test to ensure screening curve is functioning correctly
  demand = pd.Series([1,2,3,4,5])
  loadCurve, loadLevel = SCM.load_levels(demand,1,5)
  #print(loadCurve)
  #print(loadLevel)
  #print(steps)
  LL = SCM.counts(loadLevel,loadCurve)
  #print(LL)
  gen1={"Name":"gen1","FOM":1,"VOM":0,"CarbonEmission":0}
  gen2={"Name":"gen2","FOM":0,"VOM":.75,"CarbonEmission":0}
  SCM.evaluate_cost(LL,gen1,0)
  SCM.evaluate_cost(LL,gen2,0)
  mincostline, capacities, totalCost = SCM.find_caps(LL,gen1,gen2)
  if totalCost == 4.75:
     print(colored("system cost analysis worked","green"))
  else: 
     print(colored("system cost analysis fails","red"))
  if capacities.loc[4]["gen1"] == 14:
     print(colored("capacities function correctly","green"))
  else: 
     print(colored("capacities function failed","red"))

def test_Syngas():
  netGen=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
  nucCap,coalCap,FCcap,capcost,FOM,nLCOE,rLCOE,sLCOE,fixcost,varcost,NGcost,count,Demand = SCM.Syngas(netGen,Syn,10,5,5,10)

def test_carbon():
  TR = 1E299
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
  Generation = pd.read_excel('~/Documents/SCM/unittests/SWPP_Demand_and_Genearation_2020.xls')
  demand = Generation['Demand (MW)']
  Solar = Generation['Net Generation (MW) from Solar']
  Wind = Generation['Net Generation (MW) from Wind']
  maxSolar = max(Solar)                         #Find maximum solar generation, this will be our solar capacity production factor of 1
  maxWind = max(Wind)
  normalizedSolar = Solar/maxSolar        #Normalize solar production between 0 and 1, with one being the index of maxSolar
  normalizedWind = Wind/maxWind           #See normalizedSolar
  normalizedWind = normalizedWind.to_numpy().flatten()    #Flatten for output
  normalizedSolar = normalizedSolar.to_numpy().flatten()  #Flatten for output
  demand = demand.to_numpy().flatten()
  WC = 10000
  SC = 10000
  gen_wind, cost_wind = SCM.wind_cap(WC,normalizedWind)
  gen_solar, cost_solar = SCM.solar_cap(SC, normalizedSolar)
  totalGen, netGen = SCM.net_gen(demand, gen_solar, gen_wind)
  capacities, totalCost, mincostline = SCM.run2(netGen,TR,Coal,Coal30,Coal90,Nuclear,NUCHTSE,CC221,CC111,CC111_90,IC,CTLM,CTGE)
  
  

test_wind_cap()
test_solar_cap()
test_net_gen()
test_storage()
test_SCM()
test_carbon()
#test_FuelCell()
#test_Syngas()