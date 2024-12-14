%% Line Search Subroutine Test
a_values = [2, -2, 5, -5, 10, -10]; % Test cases for a
initial_guess = 0.1;                % Initial guess for lambda

% Initialize total function evaluations counter
total_N_eval = 0;

for i = 1:length(a_values)
    a = a_values(i); % Assign the current test case
    fprintf('Test %d, a = %d:\n', i, a); % Print current test info

    F = @(lambda) (1 - 10^a * lambda)^2; % Define the test function

    % Initialize test-specific function evaluation counter
    test_N_eval = 0;

    % Perform line search and track function evaluations
    % Start evaluations from 0 for this test:
    [lambda, test_N_eval] = line_search(F, initial_guess, 0); 
    % Add to the total evaluation count:
    total_N_eval = total_N_eval + test_N_eval;                

    % Calculate expected lambda
    expected_lambda = 1 / 10^a;
    error = abs(lambda - expected_lambda);
    F_lambda = F(lambda);

    % Print results
    fprintf('Computed lambda = %.5f\n', lambda);
    fprintf('Expected lambda = %.5f\n', expected_lambda);
    fprintf('Error = %.5e\n', error);
    fprintf('F(lambda) = %.5e (should be close to 0)\n', F_lambda);
    fprintf('Function evaluations in this test = %d\n\n', test_N_eval);
end

fprintf('Total function evaluations in all tests = %d\n', total_N_eval);
%% Test 1
[t,y] = data1;
[x,N_eval,N_iter,normg] = gaussnewton(@phi2,t,y,[1;2;3;4],1e-4,1,1);

%% Diverse initial guesss test
[t,y] = data1;
t1 = [0; 0; 0; 0;];         % Neutral start
t2 = [100; 100; 100; 100];  % Far-from-solution guess
t3 = [-10; 0; -10; 0;];     % Negative values 
                            % (x2 and x4 must >=0 for phi2, x2 must >= 0 for phi1)

t_choice = t3;              % Current test

[x,N_eval,N_iter,normg] = gaussnewton(@phi2,t,y,t_choice,1e-4,1,1);

%% Task - phi1, data1
[t,y] = data1;
[x,N_eval,N_iter,normg] = gaussnewton(@phi1,t,y,[1;2;3;4],1e-4,1,1);

%% Task - phi1, data2
[t,y] = data2;
[x,N_eval,N_iter,normg] = gaussnewton(@phi1,t,y,[1;2;3;4],1e-4,1,1);

%% Task - phi2, data1 (Test 1)
[t,y] = data1;
[x,N_eval,N_iter,normg] = gaussnewton(@phi2,t,y,[1;2;3;4],1e-4,1,1);

%% Task - phi2, data2
[t,y] = data2;
[x,N_eval,N_iter,normg] = gaussnewton(@phi2,t,y,[1;2;3;4],1e-4,1,1);

%% Fit phi1 to data1/data2 for multiple starting points
% Format results_phi_data: initial guesses, final values, N_eval, N_iter, normg
[t1, y1] = data1;
[t2, y2] = data2;

% Define range of starting points
num_start_points = 4;                            % Number of different starting points
x1_range = linspace(0.1, 20, num_start_points);
x2_range = linspace(0.01, 10, num_start_points); % Reminder: x2>0 required

% Loop over different starting points for data1
results_phi1_data1 = [];
fprintf('Fitting phi1 to data1:\n');
for x1 = x1_range
    for x2 = x2_range
        x0 = [x1; x2]; % Initial guess
        [x, N_eval, N_iter, normg] = gaussnewton(@phi1, t1, y1, x0, 1e-4, 0, 0);
        results_phi1_data1 = [results_phi1_data1; x0', x', N_eval, N_iter, normg];
        fprintf(['Initial: [%.2f, %.2f], Final: [%.2f, %.2f], ' ...
            'Iterations: %d, Normg: %.2e\n'], ...
            x1, x2, x(1), x(2), N_iter, normg);
    end
end

% Loop over different starting points for data2
results_phi1_data2 = [];
fprintf('\nFitting phi1 to data2:\n');
for x1 = x1_range
    for x2 = x2_range
        x0 = [x1; x2]; % Initial guess
        [x, N_eval, N_iter, normg] = gaussnewton(@phi1, t2, y2, x0, 1e-4, 0, 0);
        results_phi1_data2 = [results_phi1_data2; x0', x', N_eval, N_iter, normg];
        fprintf(['Initial: [%.2f, %.2f], Final: [%.2f, %.2f], ' ...
            'Iterations: %d, Normg: %.2e\n'], ...
            x1, x2, x(1), x(2), N_iter, normg);
    end
end

%% Fit phi2 to data1/data2 for multiple starting points
% Format results_phi_data: initial guesses, final values, N_eval, N_iter, normg
[t1, y1] = data1;
[t2, y2] = data2;

% Define the range of starting points
num_start_points = 4;                            % Number of different starting points
x1_range = linspace(0.1, 20, num_start_points);  
x2_range = linspace(0.01, 10, num_start_points); % Reminder x2>0 required
x3_range = linspace(0.1, 20, num_start_points);  
x4_range = linspace(0.01, 10, num_start_points); % Reminder: x4>0 required

% Loop over different starting points for data1
results_phi2_data1 = [];
fprintf('\nFitting phi2 to data1:\n');
for x1 = x1_range
    for x2 = x2_range
        for x3 = x3_range
            for x4 = x4_range
                x0 = [x1; x2; x3; x4]; % Initial guess
                [x, N_eval, N_iter, normg] = gaussnewton(@phi2, t1, y1, x0, 1e-4, 0, 0);
                results_phi2_data1 = [results_phi2_data1; ...
                    x0', x', N_eval, N_iter, normg];
                fprintf(['Initial: [%.2f, %.2f, %.2f, %.2f], ' ...
                    'Final: [%.2f, %.2f, %.2f, %.2f], ' ...
                    'Iterations: %d, Normg: %.2e\n'], ...
                    x0(1), x0(2), x0(3), x0(4), x(1), x(2), x(3), x(4), N_iter, normg);
            end
        end
    end
end

% Fit phi2 to data2 using the best results from phi1 on data2
fprintf('\nFitting phi2 to data2:\n');
results_phi2_data2 = [];

% Find the best x1 and x2 from phi1 on data2 (minimum normg)
[min_value, best_index] = min(results_phi1_data2(:, end));
best_x1_x2 = results_phi1_data2(best_index, 3:4); % Extract best [x1, x2]
best_x1 = best_x1_x2(1);
best_x2 = best_x1_x2(2);

% Iterate over x3 and x4 while keeping x1 and x2 fixed
for x3 = x3_range
    for x4 = x4_range
        x0 = [best_x1; best_x2; x3; x4]; % Use best x1 and x2, vary x3 and x4
        [x, N_eval, N_iter, normg] = gaussnewton(@phi2, t2, y2, x0, 1e-4, 0, 0);
        results_phi2_data2 = [results_phi2_data2; x0', x', N_eval, N_iter, normg];
        fprintf(['Initial: [%.2f, %.2f, %.2f, %.2f], ' ...
            'Final: [%.2f, %.2f, %.2f, %.2f], ' ...
            'Iterations: %d, Normg: %.2e\n'], ...
            x0(1), x0(2), x0(3), x0(4), x(1), x(2), x(3), x(4), N_iter, normg);
    end
end