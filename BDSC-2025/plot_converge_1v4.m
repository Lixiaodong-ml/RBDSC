function plot_converge_1v4(obj_func_values_D2, acc_values_D2, nmi_values_D2, ne_values_D2, scds_values_D2, exp_n, data_name, iParam, iRepeat) 
% Number of iterations
% figure;
hold off;
iterations = length(obj_func_values_D2);
x = [1:iterations]'; 

% Define custom colors and markers
objective_color = [0 0.4470 0.7410]; % A sharp blue
acc_color = [0.8500 0.3250 0.0980];  % A vibrant orange
nmi_color = [0.4660 0.6740 0.1880]; % A bright green
ne_color = [0.4940 0.1840 0.5560];  % A deep purple
scds_color = [0.9290 0.6940 0.1250]; % A bright yellow
marker_obj = 'o'; % Circle markers for objective function
marker_acc = 's'; % Square markers for ACC
marker_nmi = 'd'; % Diamond markers for NMI
marker_ne = 'v';  % Triangle markers for NE
marker_scds = '^'; % Triangle (up) markers for SCDS

% Plot objective function on the left axis
yyaxis left;
plot(x, obj_func_values_D2, 'Color', objective_color, ...
    'Marker', marker_obj, 'LineWidth', 1.5, 'MarkerSize', 1.5);
% 设置 X 轴范围
xlim([1, length(x)]);
ylabel('Objective Function Value', 'Color', objective_color); % Left Y-axis
set(gca, 'YColor', objective_color); % Set the left Y-axis color to match

% Plot ACC on the right axis
yyaxis right;
plot(x, acc_values_D2, 'Color', acc_color, ...
    'Marker', marker_acc, 'LineWidth', 1.5, 'MarkerSize', 1.5);
xlim([1, length(x)]);
ylim([0,1]);
ylabel('ACC (%)', 'Color', acc_color); % Right Y-axis
set(gca, 'YColor', acc_color); % Set the right Y-axis color to match

% Add additional plots for NMI, NE, and SCDS on the right axis
hold on; % Keep the previous plots
plot(x, nmi_values_D2, 'Color', nmi_color, ...
    'Marker', marker_nmi, 'LineWidth', 1.5, 'MarkerSize', 1.5);
xlim([1, length(x)]);
plot(x, ne_values_D2, 'Color', ne_color, ...
    'Marker', marker_ne, 'LineWidth', 1.5, 'MarkerSize', 1.5);
xlim([1, length(x)]);
plot(x, scds_values_D2, 'Color', scds_color, ...
    'Marker', marker_scds, 'LineWidth', 1.5, 'MarkerSize', 1.5);
xlim([1, length(x)]);
% Common labels and settings
xlabel('Number of Iterations');
legend({'Objective Function', 'ACC', 'NMI', 'NE', 'SCDS'}, ...
    'Location', 'best', 'AutoUpdate','off'); % Custom legend

% Ensure the box is on for both axes
box on;

% Overall adjustments
set(gcf, 'Color', 'w'); % Set figure background to white for a clean look
hold off;
pdf_dir = [exp_n, filesep, data_name];  
save2pdf(fullfile(pdf_dir, [data_name, '_Param', num2str(iParam), '_Repeat ', num2str(iRepeat), '_', exp_n, '_converge.pdf']), gcf, 600);