function plot_converge(obj_func_values_D2, acc_values_D2, exp_n, data_name, iParam, iRepeat) 
% Number of iterations
iterations = 1:length(obj_func_values_D2);

% Define custom colors and markers
objective_color = [0 0.4470 0.7410]; % A sharp blue
acc_color = [0.8500 0.3250 0.0980];  % A vibrant orange
marker_obj = '.'; % Square markers for objective function
marker_acc = '.'; % Circle markers for ACC

% % Plot for dataset D2
% [ax1, h1, h2] = plotyy(iterations, obj_func_values_D2, iterations, acc_values_D2);
% 
% xlabel('Number of iterations');
% ylabel(ax1(1), 'Objective function value'); % Left y-axis
% ylabel(ax1(2), 'ACC (%)'); % Right y-axis
% 
% % Customize the plot for better visual impact
% set(h1, 'LineWidth', 2, 'Color', objective_color, 'Marker', marker_obj, 'MarkerSize', 15); % Objective function in blue
% set(h2, 'LineWidth', 2, 'Color', acc_color, 'Marker', marker_acc, 'MarkerSize', 15); % ACC in orange
% set(ax1(1), 'YColor', objective_color); % Set the left axis color to blue
% set(ax1(2), 'YColor', acc_color);       % Set the right axis color to orange
% 
% 
% % Add legend for clarity
% legend([h1; h2], 'Objective function value', 'ACC', 'Location', 'NorthWest');

% Plot objective function on the left axis
yyaxis left;
plot(iterations, obj_func_values_D2, 'Color', objective_color, ...
    'Marker', marker_obj, 'LineWidth', 2, 'MarkerSize', 15);
% ylabel('Objective function value', 'Color', objective_color); % Left Y-axis
ylabel('目标函数值', 'Color', objective_color); % Left Y-axis
set(gca, 'YColor', objective_color); % Set the left Y-axis color to match

% Plot ACC on the right axis
yyaxis right;
plot(iterations, acc_values_D2, 'Color', acc_color, ...
    'Marker', marker_acc, 'LineWidth', 2, 'MarkerSize', 15);
ylabel('ACC', 'Color', acc_color); % Right Y-axis
set(gca, 'YColor', acc_color); % Set the right Y-axis color to match

% Common labels and settings
% xlabel('Number of iterations');
xlabel('迭代次数');
% legend('Objective function value', 'ACC', 'Location', 'east');
legend('目标函数值', 'ACC', 'Location', 'east');

% Ensure the box is on for both axes
box on;


% Overall adjustments
set(gcf, 'Color', 'w'); % Set figure background to white for a clean look
pdf_dir = [exp_n, filesep, data_name];  
save2pdf(fullfile(pdf_dir, [data_name, '_Param', num2str(iParam), '_Repeat ', num2str(iRepeat), '_', exp_n, '_converge.pdf']), gcf, 600);