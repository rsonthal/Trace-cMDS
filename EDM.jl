module EDM

using FFTW, Plots, LinearAlgebra, ProgressMeter, SparseArrays, Random, PyCall, Statistics, Distances

## The Input is the sqaured EDM and outputs the squared EDM
## If you want to embed into dim d then set k to be d+1
function euclidean_rank(D,k; tol = 1e-8, verbose = false)
    (n,n) = size(D)
    G = Symmetric(-1*copy(D))
    count = 0
    v = ones(n)
    v[n] += sqrt(n)
    Q = Hermitian(Matrix(I,n,n) - (2/norm(v)^2)*v*v')
    c = (2/norm(v)^2)
    
    err = 1
    while(err > tol && count < 5000)
        
        for i = 1:n
            G[i,i] = 0
        end
        
        F = G - c*v*(v'*G) - c*(G*v)*v' + c*c*v*(v'*((G*v)*v'))
        F̂ = Symmetric(F[1:n-1,1:n-1])

        Λ, U = eigen(-1*F̂,1:k-1)
        if count % 50 == 0 && verbose
            @show(F[n,n])
            flush(stdout)
        end
        Λ = min.(0,Λ)
        
        S = sum(Λ)-F[n,n]
        
        F[n,n] += S/k
        Λ .-= S/k
        if F[n,n] > 0
            print("NOOoOOOOOOOooo")
            flush(stdout)
        end
        
        F̂′ = -1*U*(Λ.*U')
        err = norm(F̂-F̂′)
        F[1:n-1,1:n-1] = F̂′
        G = F - c*v*(v'*F) - c*(F*v)*v' + c*c*v*(v'*((F*v)*v'))
        
        if count % 50 == 0 && verbose
            @show((count,err))
            flush(stdout)
        end
        
        count+=1
    end
    
    G = -1*G
    return G
end

## The Input is the sqaured EDM and outputs the squared EDM
## If you want to embed into dim d then set k to be d+1
## Here the output will be entry wise bigger than input. 
function euclidean_increase_rank(D,k)
    (n,n) = size(D)
    G = Symmetric(-1*copy(D))
    count = 0
    v = ones(n)
    v[n] += sqrt(n)
    Q = Hermitian(Matrix(I,n,n) - (2/norm(v)^2)*v*v')
    c = (2/norm(v)^2)
    
    Z′ = spzeros(n,n)
    err = 1
    while(err > 1e-1 && count < 5000)
        
        for i = 1:n
            G[i,i] = 0
        end
        
        F = G - c*v*(v'*G) - c*(G*v)*v' + c*c*v*(v'*((G*v)*v'))
        F̂ = Symmetric(F[1:n-1,1:n-1])

        Λ, U = eigen(-1*F̂,1:k-1)
        if count % 50 == 0
            @show(F[n,n])
            flush(stdout)
        end
        Λ = min.(0,Λ)
        
        S = sum(Λ)-F[n,n]
        
        F[n,n] += S/k
        Λ .-= S/k
        if F[n,n] > 0
            print("NOOoOOOOOOOooo")
            flush(stdout)
        end
        
        F̂′ = -1*U*(Λ.*U')
        err = norm(F̂-F̂′)
        F[1:n-1,1:n-1] = F̂′
        G = F - c*v*(v'*F) - c*(F*v)*v' + c*c*v*(v'*((F*v)*v'))
        
        if count % 50 == 0
            @show((count,err))
            flush(stdout)
        end
        
        G = -1*G
        
        for i = 1:n
            for j = 1:i-1
                if G[j,i]-D[j,i]^2 < Z′[j,i]
                    Z′[j,i] -= G[j,i]-D[j,i]^2
                    G[j,i] = D[j,i]^2 
                    G[i,j] = G[j,i]
                    
                else
                    G[j,i] -= Z′[j,i]
                    G[i,j] = G[j,i]
                    Z′[j,i] = 0
                end
            end
        end
        
        G = -1*G
        
        count+=1
    end
    
    G = -1*G
    return G
end

#Takes in the sqaured perturned EDM and returns the squared EDM
function create_lower_bound(Dp,r; verbose = false)
    (n,n) = size(Dp)
    v = ones(n)
    v[n] += sqrt(n)
    Q = Hermitian(Matrix(I,n,n) - (2/norm(v)^2)*v*v')
    
    Dphat = Q*Dp*Q
    
    Eig = eigen(Symmetric(Dphat[1:n-1,1:n-1]))
    Λ = copy(Eig.values)
    s = 0
    for i = 1:n-1 
        if Λ[i] > 0 || i > r
            s += Λ[i]
            Λ[i] = 0
        end
    end

    E = sum(Λ .!= 0) + 1
    sub = s/E

    if verbose
        @show((E,s,sub))
    end
    
    Dphat[n,n] += sub
    E -= 1
    s -= sub

    for i = 1:n-1
        if Λ[n-i] != 0
            if Λ[n-i] + sub <= 0     
                Λ[n-i] += sub
                E -= 1
                s -= sub
            else
                E -= 1
                s += Λ[n-i]
                Λ[n-i] = 0
                sub = s/E
            end
        end
    end    
    
    Dphat[1:n-1,1:n-1] = Eig.vectors*diagm(Λ)*Eig.vectors';

    if verbose
        @show(sum(Λ))
        @show(Dphat[n,n])
        @show(rank(Dphat))
    end
    
    return Q*Dphat*Q
end

function MDS_error(Dp, Dcmds, r)
    (n,n) = size(Dp)
    v = ones(n)
    v[n] += sqrt(n)
    Q = Hermitian(Matrix(I,n,n) - (2/norm(v)^2)*v*v')
    
    Dphat = -1*Q*Dp*Q
    A = Dphat[1:n-1,1:n-1]
    lambda = eigvals(Symmetric(A))
    Lambda = copy(lambda)
    
    p = sortperm(lambda, rev = true)
    for i = 1:n-1
        if lambda[p[i]] > 0 && i <= r
            Lambda[p[i]] = 0
        end
    end
    
    C1 = sum(Lambda.^2)
    C2 = (sum(Lambda)^2)
    
    B = Q*(Dp-Dcmds)*Q
    C = zeros(size(B))
    C[1:n-1,1:n-1] = B[1:n-1, 1:n-1]
    
    Eig = eigen(Symmetric(C))
    U = Q*Eig.vectors
    L = Eig.values
    
    C3 = (n) * sum(((U.*U)*L).^2)
    #C3 = n *sum(diag(Q*C*Q).^2)
    
    errvec = 2*(B[n,:])
    errvec[n] /= 2
    
    #C3 = sum(errvec.^2)
    
    error = C1 + C2/2 + C3/2
    
    return error,C1,C2,C3
end

function Kmin(D,K)
    (n,n) = size(D)
    for i = 1:n
        M = findKmin(D[i,:], K+1)
        for j = 1:n
            if D[i,j] > M
                D[i,j] = Inf
            end
        end
    end
    
    for i = 1:n
        for j = 1:n
            if D[i,j] != Inf
                D[j,i] = D[i,j]
            end
        end
    end
    
    return D
end  

function findKmin(A,k)
    A = Set(A)
    for i = 1:k-1
        m = minimum(A)
        A = delete!(A,m)
        if isempty(A)
            return m
        end
    end
    
    return minimum(A)
end

function apsp(G)
    (n,n) = size(G)
    U = Inf*ones(size(G))
    for i = 1:n
        U[i,i] = 0
    end
    
    for i = 1:n
        for j = 1:n
            if G[i,j] != Inf
                U[i,j] = G[i,j]
            end
        end
    end
    
    for k = 1:n
        for i=1:n
            for j=1:i-1
                if U[i,j] > U[i,k] + U[k,j]
                    U[i,j] = U[i,k]+U[k,j]
                    U[j,i] = U[i,j]
                end
            end
        end
    end
    
    return U
end

#Takes in sqaured perturbned matrix
function mds(D, d; center=true,align=true)
    n = size(D, 1)
    
    P = Matrix(I,n,n) - ones(n, n) / n
    K = -0.5*(P*(D)*P') 
    K = (K+K')/2

    e, V = eigen(K)
    idx = sortperm(e, rev=true)[1:d] 
    
    if (d > sum(e .> 0))
        println("Using some eigenvaectors for negative eigenvalues")
    end

    X =  V[:,idx]*diagm(0 => sqrt.(e[idx[1:d]]))

    if center == true
        X .-= sum(X,dims=1)/(size(X)[1])
    end

    if align == true
        for i =1:d
            j = argmax(abs.(X[:,i]))
            X[:,i] .*= sign.(X[j,i])
        end
    end

    return X
end

function procrustes(X, Y)
    muX = mean(X; dims=1)
    muY = mean(Y; dims=1)
    
    X0 = X .- muX 
    Y0 = Y .- muY 

    # Procrustes rotation
    U, _, V = svd(X0' * Y0; full=false) 
    Q = V * U' 

    # Optimal scaling
    alpha = tr(X0' * Y0 * Q) / tr(Y0' * Y0)  

    # Align data
    Ya = alpha * (Y0 * Q) .+ muX
    
    return Ya, Q
end


end