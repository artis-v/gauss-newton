function [lambda, temp_N_eval] = line_search(F, lambda0, N_eval, n)

    % Inputs:
    % F           - Function handle for F(lambda)
    % lambda0     - Initial guess for lambda
    % N_eval      - Function evaluation counter to be updated
    % n           - Number of parameters in x
    
    % Outputs:
    % lambda      - Step length that minimizes F(lambda)
    % temp_N_eval - Number of function evaluations within the line search

    % Parameters
    alpha = 2;         % Alpha from Alg 3, p.53
    lambda = lambda0;  % Initial lambda
    F0 = F(0);         % Initial F(lambda) value
    gf = grad(F,0);    % F'(0)
    ep = 0.1;          % Epsilon from Alg 3, p.53

    % Increment function evaluations for F(0)
    temp_N_eval = 1;

    % Increment function evaluations for F'(0)
    temp_N_eval = temp_N_eval + 2*n; % grad.m gradient evaluations require two function 
                                     % evals (xplus and xminus) for every parameter in x

    % Increase lambda while sufficient reduction is not met
    while F(alpha*lambda) < F0 + ep * gf * alpha * lambda
        lambda = alpha * lambda;
        temp_N_eval = temp_N_eval + 1;
        fprintf(['  Line Search Iteration: lambda = %.4e, ' ...
            'F(lambda) = %.4e\n'], lambda, F(lambda));
    end

    % Decrease lambda while F(lambda) is still too large
    while F(lambda) > F0 + ep * gf * lambda
        lambda = lambda / alpha;
        temp_N_eval = temp_N_eval + 1;
        fprintf(['  Line Search Backtrack Iteration: lambda = %.4e, ' ...
            'F(lambda) = %.4e\n'], lambda, F(lambda));
    end

    % Warning to handle invalid or problematic results
    if isnan(F(lambda)) || F(lambda) > F0
        warning(['Potential issue with the line search: ' ...
            'F(lambda) = %.4e, F(0) = %.4e'], F(lambda), F0);
    end

    return;
end