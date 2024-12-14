function [x, N_eval, N_iter, normg] = gaussnewton(phi, t, y, x0, tol, printout, plotout)
    tic; % Start timing

    % Inputs:
    % phi      - Function handle for the model function phi(x, t)
    % t        - Vector of independent variable data
    % y        - Vector of dependent variable data (observations)
    % x0       - Initial guess for the parameter vector x
    % tol      - Tolerance for stopping criteria
    % printout - Printing intermediate results (1: print, 0: no print)
    % plotout  - Plotting results (1: plot, 0: no plot)
    
    % Outputs:
    % x        - Final parameter estimate
    % N_eval   - Number of function evaluations
    % N_iter   - Number of iterations performed
    % normg    - Norm of the gradient at the final parameter estimate

    % 1. Initialize variables
    x = x0;         % Start with the initial guess
    N_eval = 0;     % Initialize function evaluation counter
    N_iter = 0;     % Initialize iteration counter
    max_iter = 100; % Set a maximum number of iterations to prevent infinite loops
    epsilon = 1e-6; % Regularization parameter for stability
    
    % 2. Define residual and objective function
    r = @(x) phi(x, t) - y; % Residual function: r(x) = phi(x, t) - y
    f = @(x) sum(r(x).^2);  % Objective function: f(x) = ||r(x)||^2

    % 3. Gauss-Newton loop
    while N_iter < max_iter
        N_iter = N_iter + 1; % Increment the iteration counter
        
        % 3.1. Evaluate the residual vector r(x)
        residual = r(x);
        N_eval = N_eval + 1;  % Increment function evaluation count
        
        % 3.2. Compute the Jacobian matrix J numerically
        m = length(residual); % Number of data points
        n = length(x);        % Number of parameters
        J = zeros(m, n);      % Initialize Jacobian matrix

        % Compute the Jacobian using grad.m
        for i = 1:m
            % Define scalar function for i-th residual
            scalar_ri = @(xi) phi(xi, t(i)) - y(i); % Residual for the i-th data point
            J(i, :) = grad(scalar_ri, x)';          % Use grad.m to compute its gradient
        end
        
        % 3.3. Compute search direction
        % Gauss-Newton direction - No regularization term:
        % d = -(J' * J) \ (J' * residual);                               
        % Gauss-Newton direction - With regularization term for numerical stability:
        d = -((J' * J) + epsilon * eye(size(J, 2))) \ (J' * residual);

        % 3.4. Perform line search to determine optimal step length
        fprintf('Gauss-Newton Iteration %d: Starting line search...\n', N_iter);
        F = @(lambda) f(x + lambda * d); % Define F(lambda)
        [lambda, N_eval] = line_search(F, 1, N_eval); % Call line search with init guess
                                                      % for lambda and update N_eval

        % 3.5. Update parameters
        x = x + lambda * d;

        % 3.6. Check for convergence
        if norm(lambda * d) < tol
            % Compute the maximum absolute residual
            max_residual = max(abs(residual));

            fprintf(['Gauss-Newton Iteration %d: Converged with tol %.4e,' ...
                ' max(abs(r)) %.4e.\n'], N_iter, tol, max_residual);
            break; % Exit loop if update step is below the tolerance
        end

        % Print intermediate results if printout flag is set
        if printout
            % Compute the maximum absolute residual
            max_residual = max(abs(residual));
            
            % Compute the gradient norm
            normg = norm(2 * J' * residual);
            
            % Print the iteration details
            fprintf('Gauss-Newton Iteration %d:\n', N_iter);
            fprintf('x = [%.4f, %.4f, %.4f, %.4f]\n', x);
            fprintf('max(abs(r)) = %.4e, norm(grad) = %.4e, lambda = %.4e\n', ...
                max_residual, normg, lambda);
        end
    end

    % Print final results
    normg = norm(2 * J' * residual); % Final gradient norm
    fprintf('\nFinal Results:\n');
    fprintf('x = [%.4f, %.4f, %.4f, %.4f]\n', x);
    fprintf('Final norm(grad) = %.4e\n', normg);
    fprintf('Total iterations = %d\n', N_iter); % Number of Gauss-Newton iterations
    fprintf('Total function evaluations = %d\n', N_eval); % Evaluations in Gauss-Newton 
                                                          % and line search

    elapsed_time = toc; % Stop timing
    fprintf('Total elapsed time: %.4f seconds\n', elapsed_time); % Display elapsed time

    % Plot results if requested
    if plotout
        figure; % Create a new figure
        plot(t, y, 'ro', 'MarkerSize', 6, 'DisplayName', 'Data'); % Plot data points
        hold on;
        % Plot fitted curve:
        plot(t, phi(x, t), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Fitted Curve'); 
        legend show; % Add a legend
        title('Gauss-Newton Fit');
        xlabel('t'); % Label x-axis
        ylabel('y'); % Label y-axis
        grid on; % Add grid -> better visualization
    end
end