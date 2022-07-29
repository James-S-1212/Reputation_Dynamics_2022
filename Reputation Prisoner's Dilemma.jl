using CSV
using StatsBase
using DataFrames
using LinearAlgebra
using Plots

# Determine model parameters and payoff matrix 
omega = 10
mu = 10^(-3)
b = 5
c = 1
epsilon = 10^(-3)
chi = 10^(-3)
alpha = 10^(-3)
payoff = [b-c -c; b 0] 
# Define social norm 
norm = 0


function graph_setup(samples, Zs, steps_played)
    # Determine interval of learning step rows in csv file with considered cooperation proportions 
    end_i = steps_played
    start_i = end_i - Int(floor(0.1 * end_i))
    mean_coop_indices = zeros(0)

    # Iterate over different population sizes in Zs and determine mean cooperation index for each one
    for j in 1:length(Zs)
        # Select population size for this iteration 
        Z = Zs[j]

        # Obtain 'samples' different cooperation indices for random populations of size Z
        coop_indices = zeros(0)
        for i in 1:samples
            print(i, " $Z \n")
            # Initialise population and reputations 
            init_type = rand(0:3)
            population = [init_type for i in 1:Z]
            reps = [0 for i in 1:Z]
            # Perform "steps_played" learning steps on the population and read into csv then data frame
            df = build_csv(population, reps, steps_played)

            # Extract and record cooperation index from interactions of final 10% of learning steps 
            coop_index = sum(df[start_i:end_i, 1]) / ((end_i - start_i) * 4000)
            append!(coop_indices, coop_index)
        end
        # Record mean cooperation index for this specific Z 
        append!(mean_coop_indices, mean(coop_indices))
    end
    # Return graphing data: Zs for x axis and cooperation indices for y axis 
    df_out = DataFrame(Population_Size=Zs, Cooperation_Index=mean_coop_indices)
    CSV.write("population_coop_indices_$norm.csv", df_out)
    return df_out
end


"""

    build_csv(pop, iters)

Test a population of agents 'pop' against each other using play() to produce a generational
csv data file. Do this for 'iters' iterations, each producing a different csv file. 

Input: 
    - 'population': Population array of agents with strategies 
    - 'reps': Reputation array of agent reputations 
    - 'rounds': Integer number of iterations to run the step() function for on the population  

Output: 
    - None, but produces csv file full of generational data 

"""
function build_csv(population, reps, rounds)
    pop = copy(population)
    rep = copy(reps)

    # Initialise datas frame to store populations and defections  
    df = DataFrame()
    cooperations = DataFrame()
    cooperations[!, "No. Cooperations"] = Int64[]
    # Initialise column names 
    for i in 1:length(population)
        colname = "Agent$i"
        df[!,colname] = Int64[]
    end
    push!(df, pop)

    # Perform 'rounds' learning steps on the population, and append new generation row to data frame after each step. Record
    # number of defections 
    for i in 1:rounds
        push!(cooperations, [step(pop, rep, omega, mu)])
        push!(df, pop)
    end

    # Write generational data to csv file 
    # CSV.write("reputation_gen_data.csv", df)

    # Write defection data to csv file 
    # CSV.write("cooperation_data.csv", cooperations)
    return cooperations
end


"""

    step(agents, omega, mu)

Take a population of agents, "agents", and perform one learning step on it. 

Inputs: 
    -"agents": Array of agents (Ints from 0-3) 
    -"omega": Omega model parameter, real value 
    -"mu": Mu (mutation chance) model parameter, real value 

Outputs: 
    -'cooperations': Number of total cooperative acts observed over the step (Int).  

"""
function step(agents, reps, omega, mu)
    # Select random two agents as players and record indices 
    players = sample(1:length(agents), 2, replace = false)
    focal_i = players[1]
    role_i = players[2]
    focal = agents[focal_i]
    role = agents[role_i]
    focal_sum = 0
    role_sum = 0
    rounds = 1000
    cooperations = 0

    # Run one step of learning, via 'rounds' interactions between focal/role with other agents 
    for i in 1:rounds
        pool = sample(1:length(agents), 2, replace = false)
        
        if pool[1] == focal_i || pool[2] == role_i
            focal_opp_i = pool[2]
            role_opp_i = pool[1]
        else
            focal_opp_i = pool[1]
            role_opp_i = pool[2]
        end

        # Determine actions each agent will take given the other's reputation
        focal_act = decide(focal, reps[focal_opp_i])
        focal_opp_act = decide(agents[focal_opp_i], reps[focal_i])
        role_opp_act = decide(agents[role_opp_i], reps[role_i])
        role_act = decide(role, reps[role_opp_i])

        # Determine payoff for each agent given actions 
        focal_sum = focal_sum + payoff[(focal_act + 1) + (2*focal_opp_act)]
        role_sum = role_sum + payoff[(role_act + 1) + (2*role_opp_act)]

        # Update reputations of opponents 
        focal_prior = reps[focal_i]
        role_prior = reps[role_i]
        #focal_opp_prior = reps[focal_opp_i]
        #role_opp_prior = reps[role_opp_i]
        #reps[focal_i] = update_rep(focal_act, focal_opp_prior)
        #reps[role_i] = update_rep(role_act, role_opp_prior)
        reps[focal_opp_i] = update_rep(focal_opp_act, focal_prior)
        reps[role_opp_i] = update_rep(role_opp_act, role_prior)

        # Add cooperations to total 
        if role_opp_i == focal_i && focal_opp_i == role_i 
            cooperations = cooperations + ((focal_act+1)%2) + ((role_act+1)%2) 
        else 
            cooperations = cooperations + ((focal_act+1)%2) + ((role_act+1)%2) + ((focal_opp_act+1)%2) + ((role_opp_act+1)%2) 
        end
    end
    # Calculate payoff for each player
    pi_f = focal_sum / rounds
    pi_r = role_sum / rounds

    # Calculate probability p, of focal agent adopting role model agent's strategy
    p = 1/(1 + exp(omega*(pi_f-pi_r)))

    # Focal adopts role model strategy with probability p 
    if rand(1)[1] < p 
        agents[focal_i] = role
    end 

    # Focal mutates to a random strategy instead, with probability mu
    if rand(1)[1] < mu
        agents[focal_i] = rand(0:3)
        # print(agents[focal_i])
    end 
    return Int(cooperations) 
end



"""

    decide(agent1_strat, agent2_rep)

Take the strategy of agent1, 'agent1_strat', and return the decision
for how agent1 will play against agent2 given agent2's reputation,
agent2_rep. 

Inputs: 
    -"agent1_strat": Strategy agent1 uses, integer 
    -"agent2_rep": Reputation of agent2, integer. 

Outputs: 
    -The decision for how agent1 will act: cooperate (0) or defect (1). 
     Integer. 

"""
function decide(agent1_strat, agent2_rep) 
    # Incorporate chi error of misjudging reputation 
    if rand(1)[1] < chi
        agent2_rep = Int((agent2_rep + 1) % 2)
    end
    
    if agent2_rep == 0  
        # Respond to good reputation
        act = Int(agent1_strat % 2)
    else  
        # Respond to bad reputation
        act = Int(agent1_strat % 2 + floor(agent1_strat / 2)) % 2
    end

    # Incorporate epsiolon error in action chance 
    if act == 0
        if rand(1)[1] < epsilon
            return 1
        end
    end
    return act
end


"""

    update_rep(agent1_act, agent2_rep)

Return new reputation for agent1 a given its action, "agent1_act", against an
agent with reputation "agent2_rep". 

Inputs: 
    -"agent1_act": Action of agent 1, an integer 
    -"agent2_rep": Reputation of agent 2, an integer 

Outputs: 
    -"agent1_rep": New reputation for agent 1, an integer 

"""
function update_rep(agent1_act, agent2_rep) 
    # Get bit in binary representation of norm corresponding to action against reputation 
    bit_pos = (((agent2_rep + 1) % 2) * 2) + ((agent1_act + 1) % 2) 
    agent1_rep = floor((norm % 2^(bit_pos + 1)) / 2^bit_pos) 

    # Incorporate alpha error in reporting reputation 
    if rand(1)[1] < alpha
        return agent1_rep
    end
    # Return flipped bit since convention here is 0 = good reputation and 1 = bad reputation 
    return Int((agent1_rep + 1) % 2)
end


"""

    plot_tests(iters)
Plot the csv data from file, round number against number of defectors in the population after that round, for 'iters' different omegas.

Inputs: 
    -'iters': Number of iterations run, which is the number of files to read. 

Outputs: 
    -'p1': The plot 

"""
function plot_tests(iters)
    ys = [[]]
    for j in 1:iters 
        df = DataFrame(CSV.File("data$j.csv"))
        y = [sum(df[i, 1:Z]) for i in 1:rounds]
        append!(ys, [y])
    end 
    popfirst!(ys)
    y_axis = ys[1] 
    for i in 2:iters
        y_axis = cat(y_axis, ys[i], dims=2)
    end
    y = convert(Matrix{Int}, y_axis)
    x = 1:rounds
    p1 = plot(x, y, title="Dominant Strategy Prevalence by Round, Varying Omega", lw=2, 
    label=["omega=0.1" "omega=1" "omega=10" "omega=100"], legend=:outertopright, titlefontsize=11, 
    xlab="Round Number", ylab="Number of Defectors in Population", xguidefontsize=9, yguidefontsize=9)
    savefig("population_plot")
    
    
    return (p1)
end

