clear; tic
% Specify the file name to read
filename = 'Sweep_Runs_o\sweep\1\out~inner';
% filename = 'Saved_Dispatch_Results\out_inner_FMU';
% filename = 'Saved_Dispatch_Results\out_inner_LTI';
% Specify the interval of x ticks
x_tick_interval=1;  % 2 hours for plot 24-hour result
% x_tick_interval=24; % 24 hours for plot 168-hour result

fid = fopen(filename);
% Get one line from the file
tline = fgetl(fid);
% when this line is not empty (the end of file)
time=[];
BOP_vyminmax=[];
% SES_vyminmax=[];
SES_vyminmax=[];
while ischar(tline)
    if any([startsWith(tline, "BOP ,") startsWith(tline, "SES ,")]) 
        data=nan(1,7);
        c = strsplit(tline,',');
        for i=1:numel(c)
            if strcmp(c{i},'t')
                t_temp = str2double(c{i+1});
            elseif strcmp(c{i},'vp')
                data(1)=str2double(c{i+1});
            elseif strcmp(c{i},'y1')
                data(2)=str2double(c{i+1});
            elseif strcmp(c{i},'y1min')
                data(3)=str2double(c{i+1});
            elseif strcmp(c{i},'y1max')
                data(4)=str2double(c{i+1});
            elseif strcmp(c{i},'y2')
                data(5)=str2double(c{i+1});
            elseif strcmp(c{i},'y2min')
                data(6)=str2double(c{i+1});
            elseif strcmp(c{i},'y2max')
                data(7)=str2double(c{i+1});
            end
        end
        
        if startsWith(tline, "BOP ,")
            BOP_vyminmax=[BOP_vyminmax;data];
            time = [time;t_temp];
        elseif startsWith(tline, "SES ,")
            SES_vyminmax=[SES_vyminmax;data];
%         elseif startsWith(tline, "TES ,")
%             TES_vyminmax=[TES_vyminmax;data];
        end
        
    end
%     disp(tline)
    tline = fgetl(fid);
end
fclose(fid);
%%
power_provided=BOP_vyminmax(:,1)+SES_vyminmax(:,1);
time_hour=[];power_array_hour=[];
for i=1:numel(time)
    if mod(time(i),3600)==1800
        time_hour = [time_hour; time(i)];
        power_array_hour=[power_array_hour;BOP_vyminmax(i,1) SES_vyminmax(i,1)];
    end
end
% convert output power to MW, convert output pressure to bar
for i=2:4
    BOP_vyminmax(:,i)=BOP_vyminmax(:,i)*1e-6;
    SES_vyminmax(:,i)=SES_vyminmax(:,i)*1e-6;
end
for i=5:7
    BOP_vyminmax(:,i)=BOP_vyminmax(:,i)*1e-5;
end
time_hour = time_hour/3600;
time = time/3600;
%% 1. Plot the power dispatch stack
x_label_min=floor(time(1));
x_label_max=ceil(time(end));

figure(10)
set(gcf,'Position',[100 50 600 500])
% Plot the stacked bar of power components
ba = bar(time_hour, power_array_hour, 'stacked', 'FaceColor','flat');hold on
ba(1).CData = [0 0.4470 0.7410];
ba(2).CData = [0.9290 0.6940 0.1250];
% Plot the total power provided
plot(time, power_provided,'LineWidth',3,'color','#7E2F8E');hold off
xlabel('Time (Hour)');ylabel('Power (MW)'); 
xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max)
legend('BOP Output Power','SES Output Power','Market Demand','Location','best')
% legend('BOP','TES Discharging(+)/Charging(-)','Market Demand','Location','best')
title('Contribution of each Power Source')

print('Figure_10.png','-dpng','-r300')

%% 2. Plot the explicit and implicit constraints v.s. time
figure(20)
set(gcf,'Position',[100 50 1600 900])
FontSize = 14;
for unit_idx=1:2
    % Plot the output power for all 3 units
    subplot(2,3,(unit_idx-1)*3+1)
    if unit_idx==1
        plot(time,BOP_vyminmax(:,1))
%         y_lb = min(BOP_vyminmax(:,1)); y_ub = max(BOP_vyminmax(:,1));
        y_lb = min(BOP_vyminmax(:,3)); y_ub = max(BOP_vyminmax(:,4));
        title('BOP Dispatched Power','FontSize',FontSize)
    elseif unit_idx==2
        plot(time,SES_vyminmax(:,1))
%         y_lb = min(SES_vyminmax(:,1)); y_ub = max(SES_vyminmax(:,1));
        y_lb = min(SES_vyminmax(:,3)); y_ub = max(SES_vyminmax(:,4));
        title('SES Dispatched Power','FontSize',FontSize)
    elseif unit_idx==3
%         plot(time,TES_vyminmax(:,1))
%         y_lb = min(TES_vyminmax(:,1)); y_ub = max(TES_vyminmax(:,1));
%         title({'TES Dispatched Power','Discharging(-)/Charging(+)'})
    end
    xlabel('Time (Hour)','FontSize',FontSize);ylabel('Power (MW)','FontSize',FontSize); 
    xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max);xtickangle(0)
    % TODO: Change the scale to the y1 level for BOP and SES
%     ylim([y_lb-1 y_ub+1])
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.2f')
    set(gca,'FontSize',FontSize)
    % Plot y1 and its min/max
    subplot(2,3,(unit_idx-1)*3+2)
    if unit_idx==1
        plot(time,BOP_vyminmax(:,4),'--r','LineWidth',3); hold on % y1max
        plot(time,BOP_vyminmax(:,2),'-k'); % y1
        plot(time,BOP_vyminmax(:,3),'--b','LineWidth',3); hold off %y1min
        y_lb = min(BOP_vyminmax(:,3)); y_ub = max(BOP_vyminmax(:,4));
        title("BOP Constraint 1: Output Power",'FontSize',FontSize); ylabel('Power (MW)','FontSize',FontSize); 
    elseif unit_idx==2
        plot(time,SES_vyminmax(:,4),'--r','LineWidth',3); hold on
        plot(time,SES_vyminmax(:,2),'-k');
        plot(time,SES_vyminmax(:,3),'--b','LineWidth',3); hold off
        y_lb = min(SES_vyminmax(:,3)); y_ub = max(SES_vyminmax(:,4));
%         y_lb = -5; y_ub = max(SES_vyminmax(:,4));
        title("SES Constraint 1: Output Power",'FontSize',FontSize); ylabel('Power (MW)','FontSize',FontSize); 
    elseif unit_idx==3
%         plot(time,TES_vyminmax(:,4),'--r','LineWidth',3); hold on
%         plot(time,TES_vyminmax(:,2),'-k');
%         plot(time,TES_vyminmax(:,3),'--b','LineWidth',3); hold off
%         y_lb = min(TES_vyminmax(:,3)); y_ub = max(TES_vyminmax(:,4));
%         title("TES Constraint 1: Hot Tank Level"); ylabel('Level (m)'); 
    end
    xlabel('Time (Hour)','FontSize',FontSize);
    xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max);xtickangle(0)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    legend('Upper Bound','Output #1','Lower Bound','Location','southeast','FontSize',FontSize)
    set(gca,'FontSize',FontSize)

    % Plot y2 and its min/max
    subplot(2,3,(unit_idx-1)*3+3)
    if unit_idx==1
        plot(time,BOP_vyminmax(:,7),'--r','LineWidth',3); hold on % y1max
        plot(time,BOP_vyminmax(:,5),'-k'); % y1
        plot(time,BOP_vyminmax(:,6),'--b','LineWidth',3); hold off %y1min
        y_lb = min(BOP_vyminmax(:,6)); y_ub = max(BOP_vyminmax(:,7));
        title("BOP Constraint 2: Turbine Pressure",'FontSize',FontSize); ylabel('Pressure (bar)','FontSize',FontSize); 
    elseif unit_idx==2
        plot(time,SES_vyminmax(:,7),'--r','LineWidth',3); hold on
        plot(time,SES_vyminmax(:,5),'-k');
        plot(time,SES_vyminmax(:,6),'--b','LineWidth',3); hold off
        y_lb = min(SES_vyminmax(:,6)); y_ub = max(SES_vyminmax(:,7));
        title("SES Constraint 2: Firing Temperature",'FontSize',FontSize); ylabel('Temperature (K)','FontSize',FontSize); 
    elseif unit_idx==3
%         plot(time,TES_vyminmax(:,7),'--r','LineWidth',3); hold on
%         plot(time,TES_vyminmax(:,5),'-k');
%         plot(time,TES_vyminmax(:,6),'--b','LineWidth',3); hold off
%         y_lb = min(TES_vyminmax(:,6)); y_ub = max(TES_vyminmax(:,7));
%         title("TES Constraint 2: Cold Tank Level"); ylabel('Level (m)'); 
    end
    xlabel('Time (Hour)','FontSize',FontSize);
    xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max);xtickangle(0)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    legend('Upper Bound','Output #2','Lower Bound','Location','southeast','FontSize',FontSize)
    set(gca,'FontSize',FontSize)
    
end
print('Figure_20.png','-dpng','-r300')

%% 3. Plot everything in one figure
figure(30)
set(gcf,'Position',[100 50 2240 1260])
subplot(2,4,[1 5]) 
% Plot the stacked bar of power components
ba = bar(time_hour, power_array_hour, 'stacked', 'FaceColor','flat');hold on
ba(1).CData = [0 0.4470 0.7410];
ba(2).CData = [0.9290 0.6940 0.1250];
% Plot the total power provided
plot(time, power_provided,'LineWidth',3,'color','#7E2F8E');hold off
xlabel('Time (Hour)');ylabel('Power (MW)'); 
xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max)
legend('BOP','TES Discharging(+)/Charging(-)','Market Demand','Location','best')
title('Contribution of each Power Source')

for unit_idx=1:2
    % Plot the output power for all 3 units
    subplot(2,4,(unit_idx-1)*4+2)
    if unit_idx==1
        plot(time,BOP_vyminmax(:,1))
%         y_lb = min(BOP_vyminmax(:,1)); y_ub = max(BOP_vyminmax(:,1));
        y_lb = min(BOP_vyminmax(:,3)); y_ub = max(BOP_vyminmax(:,4));
        title('BOP Dispatched Power')
    elseif unit_idx==2
        plot(time,SES_vyminmax(:,1))
%         y_lb = min(SES_vyminmax(:,1)); y_ub = max(SES_vyminmax(:,1));
        y_lb = min(SES_vyminmax(:,3)); y_ub = max(SES_vyminmax(:,4));
        title('SES Dispatched Power')
%     elseif unit_idx==3
%         plot(time,TES_vyminmax(:,1))
%         y_lb = min(TES_vyminmax(:,1)); y_ub = max(TES_vyminmax(:,1));
%         title({'TES Dispatched Power','Discharging(-)/Charging(+)'})
    end
    xlabel('Time (Hour)');ylabel('Power (MW)'); 
    xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max)
    % TODO: Change the scale to the y1 level for BOP and SES
%     ylim([y_lb-1 y_ub+1])
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.2f')
    
    % Plot y1 and its min/max
    subplot(2,4,(unit_idx-1)*4+3)
    if unit_idx==1
        plot(time,BOP_vyminmax(:,4),'--r','LineWidth',3); hold on % y1max
        plot(time,BOP_vyminmax(:,2),'-k'); % y1
        plot(time,BOP_vyminmax(:,3),'--b','LineWidth',3); hold off %y1min
        y_lb = min(BOP_vyminmax(:,3)); y_ub = max(BOP_vyminmax(:,4));
        title("BOP Constraint 1: Output Power"); ylabel('Power (MW)'); 
    elseif unit_idx==2
        plot(time,SES_vyminmax(:,4),'--r','LineWidth',3); hold on
        plot(time,SES_vyminmax(:,2),'-k');
        plot(time,SES_vyminmax(:,3),'--b','LineWidth',3); hold off
        y_lb = min(SES_vyminmax(:,3)); y_ub = max(SES_vyminmax(:,4));
%         y_lb = -5; y_ub = max(SES_vyminmax(:,4));
        title("SES Constraint 1: Output Power"); ylabel('Power (MW)'); 
%     elseif unit_idx==3
%         plot(time,TES_vyminmax(:,4),'--r','LineWidth',3); hold on
%         plot(time,TES_vyminmax(:,2),'-k');
%         plot(time,TES_vyminmax(:,3),'--b','LineWidth',3); hold off
%         y_lb = min(TES_vyminmax(:,3)); y_ub = max(TES_vyminmax(:,4));
%         title("TES Constraint 1: Hot Tank Level"); ylabel('Level (m)'); 
    end
    xlabel('Time (Hour)');xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    legend('Upper Bound','Output #1','Lower Bound','Location','best')
    % Plot y2 and its min/max
    subplot(2,4,(unit_idx-1)*4+4)
    if unit_idx==1
        plot(time,BOP_vyminmax(:,7),'--r','LineWidth',3); hold on % y1max
        plot(time,BOP_vyminmax(:,5),'-k'); % y1
        plot(time,BOP_vyminmax(:,6),'--b','LineWidth',3); hold off %y1min
        y_lb = min(BOP_vyminmax(:,6)); y_ub = max(BOP_vyminmax(:,7));
        title("BOP Constraint 2: Turbine Pressure"); ylabel('Pressure (bar)'); 
    elseif unit_idx==2
        plot(time,SES_vyminmax(:,7),'--r','LineWidth',3); hold on
        plot(time,SES_vyminmax(:,5),'-k');
        plot(time,SES_vyminmax(:,6),'--b','LineWidth',3); hold off
        y_lb = min(SES_vyminmax(:,6)); y_ub = max(SES_vyminmax(:,7));
        title("SES Constraint 2: Firing Temperature"); ylabel('Temperature (K)'); 
%     elseif unit_idx==3
%         plot(time,TES_vyminmax(:,7),'--r','LineWidth',3); hold on
%         plot(time,TES_vyminmax(:,5),'-k');
%         plot(time,TES_vyminmax(:,6),'--b','LineWidth',3); hold off
%         y_lb = min(TES_vyminmax(:,6)); y_ub = max(TES_vyminmax(:,7));
%         title("TES Constraint 2: Cold Tank Level"); ylabel('Level (m)'); 
    end
    xlabel('Time (Hour)');xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    legend('Upper Bound','Output #2','Lower Bound','Location','best')
        
    
end
print('Figure_30.png','-dpng','-r300')

%% 4. Plot the implicit constraints in 2D 
figure(40)
set(gcf,'Position',[100 100 1400 600])

for unit_idx=1:3
    % Plot y1(x) with y2 (y) and their min/max
    subplot(1,3,unit_idx)
    if unit_idx==1 % BOP
        x_lb = min(BOP_vyminmax(:,3)); x_ub = max(BOP_vyminmax(:,4));
        y_lb = min(BOP_vyminmax(:,6)); y_ub = max(BOP_vyminmax(:,7));
        rectangle('Position',[x_lb y_lb x_ub-x_lb y_ub-y_lb],'LineStyle','--','LineWidth',3)
        xlim([x_lb-(x_ub-x_lb)*0.2 x_ub+(x_ub-x_lb)*0.2])
        ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
        hold on
        plot(BOP_vyminmax(:,2),BOP_vyminmax(:,5),'-o')
        hold off        
        title("BOP Implicit Constraints"); 
        ylabel('y2, Turbine Pressure (bar)'); 
        xlabel('y1, Output Power (MW)')
%         legend('Turbine Pressure v.s. Output Power')
    elseif unit_idx==2
        x_lb = min(SES_vyminmax(:,3)); x_ub = max(SES_vyminmax(:,4));
        y_lb = min(SES_vyminmax(:,6)); y_ub = max(SES_vyminmax(:,7));
        rectangle('Position',[x_lb y_lb x_ub-x_lb y_ub-y_lb],'LineStyle','--','LineWidth',3)
        xlim([x_lb-(x_ub-x_lb)*0.2 x_ub+(x_ub-x_lb)*0.2])
        ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
        hold on
        plot(SES_vyminmax(:,2),SES_vyminmax(:,5),'-o')
        hold off
        title("SES Implicit Constraints"); 
        ylabel('y2, Firing Temperature (K)'); 
        xlabel('y1, Output Power (MW)')
%         legend('Firing Temperature v.s. Output Power')
    elseif unit_idx==3
%         x_lb = min(TES_vyminmax(:,3)); x_ub = max(TES_vyminmax(:,4));
%         y_lb = min(TES_vyminmax(:,6)); y_ub = max(TES_vyminmax(:,7));
%         rectangle('Position',[x_lb y_lb x_ub-x_lb y_ub-y_lb],'LineStyle','--','LineWidth',3)
%         xlim([x_lb-(x_ub-x_lb)*0.2 x_ub+(x_ub-x_lb)*0.2])
%         ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
%         hold on
%         plot(TES_vyminmax(:,2),TES_vyminmax(:,5),'-o')
%         hold off
%         title("TES Implicit Constraints"); 
%         ylabel('y2, Cold Tank Level (m)') 
%         xlabel('y1, Hot Tank Level (m)')
% %         legend('Cold Tank Level v.s. Hot Tank Level')
    end
end
print('Figure_40.png','-dpng','-r300')


%% 5. Plot the r,v,y of BOP, in the format of "Self Learning Stage" and "Dispatching Stage" 
figure(50)
set(gcf,'Position',[100 100 1400 900])
FontSize = 15;
% define the logical arrays for learning and dispatching stages
t_learn = time<0; x_learn_min = floor(min(time(t_learn))); x_learn_max = ceil(max(time(t_learn)));
t_dispa = time>0; x_dispa_min = floor(min(time(t_dispa))); x_dispa_max = ceil(max(time(t_dispa)));
% plot the input and outputs during learning and dispatching stages
col_plot=9;
for row_idx=1:3
    % plot self-learning stage
    subplot(3,col_plot,(row_idx-1)*col_plot+[1:2])
    hold on
    if row_idx==1 % power setpoint
        plot(time(t_learn),BOP_vyminmax(t_learn,1),'Color','#0072BD')
        ylabel('Power Setpoint (MW)','FontSize',FontSize); 
        y_lb = min(BOP_vyminmax(:,1)); y_ub = max(BOP_vyminmax(:,1));
%         title("BOP, 2-hour Self-learning Stage",'FontSize',FontSize)
    elseif row_idx==2 % power output
        plot(time(t_learn),BOP_vyminmax(t_learn,4),'--r','LineWidth',3)
        plot(time(t_learn),BOP_vyminmax(t_learn,2),'-k')
        plot(time(t_learn),BOP_vyminmax(t_learn,3),'--b','LineWidth',3)
        ylabel('Power output (MW)','FontSize',FontSize);
        y_lb = min(BOP_vyminmax(:,3)); y_ub = max(BOP_vyminmax(:,4));
        legend('Upper Bound','Output #1','Lower Bound','Location','best','FontSize',FontSize)
    elseif row_idx==3 % pressure
        plot(time(t_learn),BOP_vyminmax(t_learn,7),'--r','LineWidth',3)
        plot(time(t_learn),BOP_vyminmax(t_learn,5),'-k')
        plot(time(t_learn),BOP_vyminmax(t_learn,6),'--b','LineWidth',3)
        ylabel('Turbine Pressure (bar)','FontSize',FontSize);
        y_lb = min(BOP_vyminmax(:,6)); y_ub = max(BOP_vyminmax(:,7));
        legend('Upper Bound','Output #2','Lower Bound','Location','best','FontSize',FontSize)
    end
    xlabel('Time (Hour)','FontSize',FontSize);
    xlim([x_learn_min x_learn_max]);xticks(x_learn_min:x_tick_interval:x_learn_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)

    % plot dispatching stage
    subplot(3,col_plot,(row_idx-1)*col_plot+[4:col_plot])
    hold on
    if row_idx==1 % power setpoint
        plot(time(t_dispa),BOP_vyminmax(t_dispa,1),'Color','#0072BD')
        ylabel('Power Setpoint (MW)','FontSize',FontSize); 
        y_lb = min(BOP_vyminmax(:,1)); y_ub = max(BOP_vyminmax(:,1));
%         title("BOP, 12-hour Dispatching Stage",'FontSize',FontSize)
    elseif row_idx==2 % power output
        plot(time(t_dispa),BOP_vyminmax(t_dispa,4),'--r','LineWidth',3)
        plot(time(t_dispa),BOP_vyminmax(t_dispa,2),'-k')
        plot(time(t_dispa),BOP_vyminmax(t_dispa,3),'--b','LineWidth',3)
        ylabel('Power output (MW)','FontSize',FontSize);
        y_lb = min(BOP_vyminmax(:,3)); y_ub = max(BOP_vyminmax(:,4));
        legend('Upper Bound','Output #1','Lower Bound','Location','best','FontSize',FontSize)
    elseif row_idx==3 % pressure
        plot(time(t_dispa),BOP_vyminmax(t_dispa,7),'--r','LineWidth',3)
        plot(time(t_dispa),BOP_vyminmax(t_dispa,5),'-k')
        plot(time(t_dispa),BOP_vyminmax(t_dispa,6),'--b','LineWidth',3)
        ylabel('Turbine Pressure (bar)','FontSize',FontSize);
        y_lb = min(BOP_vyminmax(:,6)); y_ub = max(BOP_vyminmax(:,7));
        legend('Upper Bound','Output #2','Lower Bound','Location','best','FontSize',FontSize)
    end
    xlabel('Time (Hour)','FontSize',FontSize);
    xlim([x_dispa_min x_dispa_max]);xticks(x_dispa_min:x_tick_interval:x_dispa_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)
end
print('Figure_50_BOP_LearningDispatching_Stage.png','-dpng','-r300')


%% 6. Plot the r,v,y of SES, in the format of "Self Learning Stage" and "Dispatching Stage" 
figure(60)
set(gcf,'Position',[100 100 1400 900])
FontSize = 15;
% define the logical arrays for learning and dispatching stages
t_learn = time<0; x_learn_min = floor(min(time(t_learn))); x_learn_max = ceil(max(time(t_learn)));
t_dispa = time>0; x_dispa_min = floor(min(time(t_dispa))); x_dispa_max = ceil(max(time(t_dispa)));
% plot the input and outputs during learning and dispatching stages
col_plot=9;
for row_idx=1:3
    % plot self-learning stage
    subplot(3,col_plot,(row_idx-1)*col_plot+[1:2])
    hold on
    if row_idx==1 % power setpoint
        plot(time(t_learn),SES_vyminmax(t_learn,1),'Color','#0072BD')
        ylabel('Power Setpoint (MW)','FontSize',FontSize); 
        y_lb = min(SES_vyminmax(:,1)); y_ub = max(SES_vyminmax(:,1));
%         title("SES, 2-hour Self-learning Stage",'FontSize',FontSize)
    elseif row_idx==2 % power output
        plot(time(t_learn),SES_vyminmax(t_learn,4),'--r','LineWidth',3)
        plot(time(t_learn),SES_vyminmax(t_learn,2),'-k')
        plot(time(t_learn),SES_vyminmax(t_learn,3),'--b','LineWidth',3)
        ylabel('Power output (MW)','FontSize',FontSize);
        y_lb = min(SES_vyminmax(:,3)); y_ub = max(SES_vyminmax(:,4));
        legend('Upper Bound','Output #1','Lower Bound','Location','best','FontSize',FontSize)
    elseif row_idx==3 % pressure
        plot(time(t_learn),SES_vyminmax(t_learn,7),'--r','LineWidth',3)
        plot(time(t_learn),SES_vyminmax(t_learn,5),'-k')
        plot(time(t_learn),SES_vyminmax(t_learn,6),'--b','LineWidth',3)
        ylabel('Firing Temp. (K)','FontSize',FontSize);
        y_lb = min(SES_vyminmax(:,6)); y_ub = max(SES_vyminmax(:,7));
        legend('Upper Bound','Output #2','Lower Bound','Location','best','FontSize',FontSize)
    end
    xlabel('Time (Hour)','FontSize',FontSize);
    xlim([x_learn_min x_learn_max]);xticks(x_learn_min:x_tick_interval:x_learn_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)

    % plot dispatching stage
    subplot(3,col_plot,(row_idx-1)*col_plot+[4:col_plot])
    hold on
    if row_idx==1 % power setpoint
        plot(time(t_dispa),SES_vyminmax(t_dispa,1),'Color','#0072BD')
        ylabel('Power Setpoint (MW)','FontSize',FontSize); 
        y_lb = min(SES_vyminmax(:,1)); y_ub = max(SES_vyminmax(:,1));
%         title("SES, 12-hour Dispatching Stage",'FontSize',FontSize)
    elseif row_idx==2 % power output
        plot(time(t_dispa),SES_vyminmax(t_dispa,4),'--r','LineWidth',3)
        plot(time(t_dispa),SES_vyminmax(t_dispa,2),'-k')
        plot(time(t_dispa),SES_vyminmax(t_dispa,3),'--b','LineWidth',3)
        ylabel('Power output (MW)','FontSize',FontSize);
        y_lb = min(SES_vyminmax(:,3)); y_ub = max(SES_vyminmax(:,4));
        legend('Upper Bound','Output #1','Lower Bound','Location','best','FontSize',FontSize)
    elseif row_idx==3 % pressure
        plot(time(t_dispa),SES_vyminmax(t_dispa,7),'--r','LineWidth',3)
        plot(time(t_dispa),SES_vyminmax(t_dispa,5),'-k')
        plot(time(t_dispa),SES_vyminmax(t_dispa,6),'--b','LineWidth',3)
        ylabel('Firing Temp. (K)','FontSize',FontSize);
        y_lb = min(SES_vyminmax(:,6)); y_ub = max(SES_vyminmax(:,7));
        legend('Upper Bound','Output #2','Lower Bound','Location','best','FontSize',FontSize)
    end
    xlabel('Time (Hour)','FontSize',FontSize);
    xlim([x_dispa_min x_dispa_max]);xticks(x_dispa_min:x_tick_interval:x_dispa_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)
end
print('Figure_60_SES_LearningDispatching_Stage.png','-dpng','-r300')

%% Plot the power dispatch stack in learning and dispatching stage
figure(70)
set(gcf,'Position',[100 100 1400 900])
FontSize = 15;
col_plot=9;
% plot the stacked bars of power components, time in hours

t_learn = time_hour<0; x_learn_min = floor(min(time_hour(t_learn))); x_learn_max = ceil(max(time_hour(t_learn)));
t_dispa = time_hour>0; x_dispa_min = floor(min(time_hour(t_dispa))); x_dispa_max = ceil(max(time_hour(t_dispa)));

subplot(1,col_plot,[1:2]) % learning
ba = bar(time_hour(t_learn), power_array_hour(t_learn,:), 'stacked', 'FaceColor','flat');hold on
ba(1).CData = [0 0.4470 0.7410];
ba(2).CData = [0.9290 0.6940 0.1250];
set(gca,'FontSize',FontSize)

subplot(1,col_plot,[4:col_plot]) % dispatching
ba = bar(time_hour(t_dispa), power_array_hour(t_dispa,:), 'stacked', 'FaceColor','flat');hold on
ba(1).CData = [0 0.4470 0.7410];
ba(2).CData = [0.9290 0.6940 0.1250];
set(gca,'FontSize',FontSize)

% Plot the line showing total power provided, time in seconds
t_learn = time<0; x_learn_min = floor(min(time(t_learn))); x_learn_max = ceil(max(time(t_learn)));
t_dispa = time>0; x_dispa_min = floor(min(time(t_dispa))); x_dispa_max = ceil(max(time(t_dispa)));

subplot(1,col_plot,[1:2]) % learning
plot(time(t_learn), power_provided(t_learn),'LineWidth',3,'color','#7E2F8E');hold off
xlabel('Time (Hour)','FontSize',FontSize);ylabel('Power (MW)','FontSize',18); 
xlim([x_learn_min x_learn_max]);xticks(x_learn_min:x_tick_interval:x_learn_max)
legend('BOP Output Power','SES Output Power','Total','Location','best','FontSize',FontSize)
title('BOP & SES, 2-hour Self-learning Stage','FontSize',FontSize)
% ylim([600 1200])

subplot(1,col_plot,[4:col_plot]) % dispatching
plot(time(t_dispa), power_provided(t_dispa),'LineWidth',3,'color','#7E2F8E');hold off
xlabel('Time (Hour)','FontSize',FontSize);ylabel('Power (MW)','FontSize',18); 
xlim([x_dispa_min x_dispa_max]);xticks(x_dispa_min:x_tick_interval:x_dispa_max)
legend('BOP Output Power','SES Output Power','Market Demand','Location','best','FontSize',FontSize)
title('BOP & SES, 12-hour Dispatching Stage','FontSize',FontSize)

print('Figure_70_Contribution_LearningDispatching.png','-dpng','-r300')





%%
toc
