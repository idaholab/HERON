from crontab import CronTab
import os
path = os.path.dirname(os.path.abspath(__file__)) 
cron = CronTab(user=True)
#md =#'cd'+" "+ path + " "+ '&&'+ " "+'/usr/bin/python'+ " " +path+'/version.py'
#print("This is cmd",cmd)
cmd = '* * * * *'+ " " +path+'/python version.py'
print ("This is cmd", cmd)
job = cron.new(command='* * * * *'+ " " +path+'/python version.py')
job.minute.every(1)
cron.write()