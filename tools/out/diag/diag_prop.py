"""Simulate inject+propagate magnitudes for open flat ground vs a colored wall,
to see whether GI 'radiance' is a visible fraction of direct sun at the surface."""
import numpy as np, math

# --- params from config / shaders ---
sun = np.array([3.2,3.1,2.8])      # sun radiance noon-ish
sky = np.array([0.45,0.55,0.75])   # sky ambient
bounce = 0.7
decay0 = math.exp(-0.5/4.0)        # cascade0
albedo_dirt = np.array([0.30,0.24,0.16])  # approx linear dirt albedo
albedo_red  = np.array([0.78,0.06,0.05])

print("decay0=%.4f reach~%.1f cells"%(decay0, 1/(1-decay0)))

# ---------- Case 1: open flat ground ----------
# 1D column, cell 0 = solid ground (top voxel), cells 1..N air above.
N=40
geom_solid = np.zeros(N,bool); geom_solid[0]=True
alb = np.tile(albedo_dirt,(N,1))
direct = np.zeros((N,3))
# air cells: skylight (skyVis=1 in open) ; the air cell directly above ground
# also gets first bounce off the ground neighbor below.
for i in range(N):
    if geom_solid[i]:
        direct[i]=0  # ground holds emission only (none)
    else:
        direct[i]=sky*1.0
        # neighbor below (i-1) solid -> first bounce. normal of ground = +Z = toward sun-ish.
        if i-1>=0 and geom_solid[i-1]:
            nrm=np.array([0,0,1.0]); sundir=np.array([0.3,0.2,0.93])
            lam=max(np.dot(nrm,sundir),0)
            direct[i]+= albedo_dirt*(sun*(1.0*lam))*(bounce*0.45)
rad=direct.copy()
# many iters to converge
for it in range(200):
    nxt=rad.copy()
    for i in range(N):
        if geom_solid[i]:
            nxt[i]=direct[i]; continue
        s=np.zeros(3)
        for d in (-1,1):
            j=i+d
            if j<0 or j>=N:
                s+=rad[i]
            elif geom_solid[j]:
                s+=rad[i]*alb[j]*bounce
            else:
                s+=rad[j]
        # 6-neighbor in real shader: 4 lateral neighbors are same air (open) -> +rad[i]
        s+=4*rad[i]
        nxt[i]=direct[i]*(1-decay0)+(decay0/6.0)*s
    rad=nxt
# surface samples the air cell just above ground (probe hops +n*0.75cell)
gi_ground = rad[1]
print("\n[OPEN GROUND] direct sun at surface = %.3f"%(sun[1]*0.93))
print("  GI radiance sampled above ground   = (%.3f,%.3f,%.3f)"%tuple(gi_ground))
print("  -> base*(direct+radiance): GI adds %.1f%% on top of direct"
      %(100*gi_ground[1]/(sun[1]*0.93)))
print("  (this 'GI' is essentially just skylight ambient ~%.2f; no colored bounce on flat open ground)"%sky[1])

# ---------- Case 2: air cell beside a sunlit RED wall (cornell) ----------
print("\n[RED WALL BOUNCE] one air cell with a red solid neighbor lit by sun:")
nrm=np.array([0,-1.0,0]); sundir=np.array([0.3,0.2,0.93])
lam=max(np.dot(nrm,sundir),0)
firstbounce = albedo_red*(sun*(1.0*lam))*(bounce*0.45)
print("  first-bounce injected into adjacent air = (%.3f,%.3f,%.3f) lam=%.2f"%(firstbounce[0],firstbounce[1],firstbounce[2],lam))
print("  note: if wall faces away from sun (interior, sun blocked) lam may be ~0 -> NO bounce injected.")
