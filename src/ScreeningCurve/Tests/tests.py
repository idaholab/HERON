#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import math
import copy
from termcolor import colored
import SCMnoStorage as SCM

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
    netGennc, storageCost,compressorSize,turbineSize,capacityStorage = SCM.Storage(netGen1,1,1,100,1,HC,HD)
    if compressorSize == 10:
       print(colored("CAES compressor size worked","green"))
    if compressorSize != 10:
       print(colored("CAES compressor size failed","red"))
    if turbineSize == 25:
       print(colored("CAES turbine size worked","green"))
    if turbineSize !=25:
       print(colored("CAES turbine size failed","red"))
    f = np.linspace(0,365,num=366)
    cost = 105.6107158535297*1000*100 + 3.00*(10*10 + 4*25)*len(f)
    if cost == storageCost:
       print(colored("CAES cost works","green"))
    if cost != storageCost:
       print(colored("CAES cost fails","red"))
    testGen = [11,12,13,14,15,16,17,18,19,20,11,12,13,14,15,16,17,18,19,20,0,0,0,0]
    testnetgen = testGen - netGennc    
    testnetgen = testnetgen.sum()
    if testnetgen == 0:
       print(colored("CAES netGen functions correctly with no curtailment","green"))
    if testnetgen != 0:
       print(colored("CAES netGen does not work with no curtailment","red"))
    netgen2 = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    netGenpeak, storageCost, compressorSize,turbineSize,capacityStorage = SCM.Storage(netgen2,1,1,100,1,HC,HD)
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
  demand = pd.DataFrame([1,2,3,4,5])
  loadCurve, loadLevel, steps = SCM.load_levels(demand,1,5)
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


test_wind_cap()
test_solar_cap()
test_net_gen()
test_storage()
test_SCM()
#test_FuelCell()
#test_Syngas()