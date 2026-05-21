module SolarDynamo

export sn, summary_statistics, hann_window

using DelayDiffEq: SDDEProblem, solve
using StochasticDiffEq: EM
using SpecialFunctions: erf
using StaticArrays
import FFTW


# --- Nonlinear function

ftilde(x, Bmin, Bmax) = x/4 * (1 + erf(x^2-Bmin^2)) * (1 - erf(x^2-Bmax^2))


# ---------------------------------
# B-field WITHOUT Jupiter
# ---------------------------------
# See here for the SDDE interface: https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/

# --- Model : B field

function f(u,h,p,t)     # Drift function
    #  u = [B, dB/dt]
    # du = [dB/dt, d^2B/dt^2]
    τ, T, Nd, sigma, Bmax = p
    hist = h(p, t - T, idxs = 1)    # B[1](t-T)
    inv_τ_2 = inv(τ^2)
    du1 = u[2]
    du2 = -u[1]*inv_τ_2 - 2*u[2]/τ - Nd*inv_τ_2 * ftilde(hist, 1, Bmax)
    SA[du1, du2]
end

function g(u,h,p,t)     # Diffusion function
    τ, T, Nd, sigma, Bmax = p
    du1 = 0
    du2 = Bmax*sigma / (τ * sqrt(τ))
    SA[du1, du2]
end


function bfield(θ, Tsim; kwargs...)

    τ, T, Nd, sigma, Bmax = θ

    # --- define initial values
    u0 = SA[Bmax, 0.0]
    # define inital values [B(t), dB(t)/dt] if t < t0
    # h(p, t) = (Bmax, 0.0)
    # we can speed things up by providing a call by index, see example linked above:
    h(p, t; idxs = nothing) = idxs == 1 ? Bmax : (Bmax, 0.0)

    # use constant lags
    lags = (T, )
    tspan = (0.0, Tsim)

    prob = SDDEProblem(f, g, u0, h, tspan, θ; constant_lags = lags)
    solve(prob, EM(); dt=0.1, saveat=1.0, kwargs...)
end


"""
```
sn(θ; Tobs = 929, Twarmup = 200, kwargs...)
```

Stochastic simulations of the number of sunspots

### Arguments
- `θ`: parameter vector `[τ, T, Nd, sigma, Bmax]`
- `Tobs`: length of the output
- `Twarmup`: length of the warm up period
- `kwargs...`: keyword arguments pased to `solve`. Mostly used ot pass a seed for the random number generator (`seed = 123`).
"""
function sn(θ; Twarmup = 200, Tobs = 929, kwargs...)

    Tsim = Twarmup + Tobs  # Total simulation steps

    sol = bfield(θ, Tsim; kwargs...)

    # square result and get rid of warm up points
    y = map(abs2, sol[1, (Twarmup + 2):end])

    return y
end


function sn_from_noise(theta, eps_dt; Twarmup=200, Tobs=929, dt=0.1, saveat=1.0)
    @assert abs(saveat - 1.0) < 1e-12 "This implementation assumes saveat == 1.0"
    @assert dt > 0

    τ, T, Nd, sigma, Bmax = theta

    Tsim = Twarmup + Tobs

    # dt-grid increments
    Ndt = Int(round(Tsim / dt))
    @assert abs(Ndt*dt - Tsim) < 1e-9 "Tsim must be multiple of dt"
    @assert length(eps_dt) >= Ndt "eps_dt too short: need Ndt = Tsim/dt"

    # delay in dt steps
    lag_steps = Int(round(T / dt))
    @assert lag_steps >= 1 "T/dt too small or dt too large"
    @assert abs(lag_steps*dt - T) < 1e-6 "T must be (approximately) a multiple of dt for this discretization"

    # EM noise scale
    coeff = Bmax * sigma / (τ^(3/2))
    sdt = sqrt(dt)

    # save every 1.0 time unit => k substeps per saved point
    k = Int(round(1.0 / dt))
    @assert abs(k*dt - 1.0) < 1e-12 "dt must divide 1.0 when saveat==1.0"

    # total saved points over [0, Tsim]: 0,1,2,...,Tsim
    Nsave = Int(round(Tsim)) + 1
    @assert abs(Tsim - (Nsave - 1)) < 1e-9 "Tsim must be integer when saveat==1.0"

    # state
    B  = Bmax
    dB = 0.0

    # ---- Ring buffer for delayed B (O(1), no popfirst!) ----
    # We store past B values at dt-grid points. The "delayed" value used at a step
    # is Bhist[hidx], where hidx points to the value from T units ago.
    Bhist = fill(Bmax, lag_steps)   # length = lag_steps
    hidx  = 1                       # next "delayed" slot to read/overwrite

    # output on integer-time grid (since saveat==1.0)
    y_save = Vector{Float64}(undef, Nsave)
    y_save[1] = B^2

    i = 0  # index into eps_dt
    @inbounds for j in 1:(Nsave-1)
        # advance by 1.0 time unit = k EM substeps
        for _sub in 1:k
            i += 1

            # delayed value (T in the past, discretized)
            B_delay = Bhist[hidx]

            # drift (same form as your SDDE definition)
            du1 = dB
            du2 = -B/τ^2 - 2*dB/τ - (Nd/τ^2) * ftilde(B_delay, 1, Bmax)

            # EM update
            dB_new = dB + du2*dt + coeff*sdt*eps_dt[i]
            B_new  = B  + du1*dt

            # update ring buffer with the NEW B (at the new time)
            Bhist[hidx] = B_new
            hidx += 1
            if hidx > lag_steps
                hidx = 1
            end

            B, dB = B_new, dB_new
        end

        y_save[j+1] = B^2
    end

    # crop warmup (integer-time indexing, consistent with your original sn)
    start = Twarmup + 2
    stop  = start + Tobs - 1
    @assert stop <= length(y_save)

    return y_save[start:stop]
end



# -------------
# Summary statistics
# -------------


"""
`hann_window(Tmax)`

Generate a Hann window of size `Tmax`.
"""
hann_window(Tmax) = [0.5*(1 - cos(2.0*π*(t-1)/(Tmax-1))) for t in 1:Tmax]


"""
`summary_statistics(data, window; fourier_range=1:6:120)`

Compute summary statistics for the input data based on the Fourier transform using a given window.

## Arguments

- `data`: The input time series data.
- `window`: Vector of windowing weights to be applied to the data. Defaults to `hann_window(length(data))`.
- `fourier_range`: Indices of the Fourier-transformed components to include in as summary statistics. Defaults to `1:6:120`.
"""
function summary_statistics(data, window=hann_window(length(data));
                            fourier_range=1:6:120)
    fs = FFTW.ifft(window .* data)
    ss = abs.(fs[fourier_range])
    return ss
end


end
