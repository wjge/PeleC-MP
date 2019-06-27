
#include <SprayParticles.H>
#include <AMReX_Particles_F.H>
#include <Transport_F.H>
#include <drag_F.H>

#include "PeleC.H"
#include "AMReX_DistributionMapping.H"
#include <AMReX_LoadBalanceKD.H>

using namespace amrex;

void
SprayParticleContainer::init_bcs()
{
    for (int dir = 0; dir < BL_SPACEDIM; dir++)
    {
       if (phys_bc->lo(dir) == Symmetry   ||
           phys_bc->lo(dir) == SlipWall   ||
           phys_bc->lo(dir) == NoSlipWall)
       {
          reflect_lo[dir] = 1;
       } else {
          reflect_lo[dir] = 0;
       }
       if (phys_bc->hi(dir) == Symmetry   ||
           phys_bc->hi(dir) == SlipWall   ||
           phys_bc->hi(dir) == NoSlipWall)
       {
          reflect_hi[dir] = 1;
       } else {
          reflect_hi[dir] = 0;
       }
    }
}

void
SprayParticleContainer::SetAll (Real val, int pstate_idx, int lev)
{
    BL_ASSERT(lev >= 0 && lev < GetParticles().size());

    ParticleLevel& plev = GetParticles(lev);

    for (auto& kv : plev)
    {
        AoS& particles = kv.second.GetArrayOfStructs();

	for (auto& p : particles)
        {
            if (p.id() > 0)
            {
                p.rdata(pstate_idx) = val;
            }
        }
    }

    return;
}

void
SprayParticleContainer::moveKickDrift (MultiFab& state,
                                      MultiFab& source,
                                      int       lev,
                                      Real      dt,
                                      int       tmp_src_width,
                                      int       where_width)
{
    const Real* dx = Geom(lev).CellSize();
    ParticleLevel&    pmap          = this->GetParticles(lev);

    BL_PROFILE("ParticleContainer::moveKickDrift()");
    BL_ASSERT(lev >= 0);
    //std::cout << " GROW -1" << state.nGrow() << std::endl;
    BL_ASSERT(state.nGrow() >= 2);

    //If there are no particles at this level
    if (lev >= this->GetParticles().size())
        return;

    const Real strttime      = ParallelDescriptor::second();

    MultiFab* state_ptr;
    MultiFab* tmp_src_ptr;

    // ********************************************************************************
    // We only make a new state_ptr if the boxArray of the state differs from the
    // boxArray onto which the particles are decomposed
    // ********************************************************************************

    if (this->OnSameGrids(lev, state))
    {
        state_ptr = &state;
    }
    else
    {
        state_ptr = new MultiFab(this->m_gdb->ParticleBoxArray(lev),
                                        this->m_gdb->ParticleDistributionMap(lev),
                                        state.nComp(), state.nGrow());
        state_ptr->setVal(0.);
        state_ptr->copy(state,0,0,state.nComp());
        state_ptr->FillBoundary(Geom(lev).periodicity()); 
    }

    // ********************************************************************************
    // We make a temporary MultiFab for the source here and initialize it to zero
    // because if we use one that already has values in it, the SumBoundary call
    // will end up duplicating those values with each call
    // 
    // ********************************************************************************
    tmp_src_width = 3; // HACK?

    tmp_src_ptr = new MultiFab(this->m_gdb->ParticleBoxArray(lev),
                                      this->m_gdb->ParticleDistributionMap(lev),
                                      source.nComp(),tmp_src_width);
    tmp_src_ptr->setVal(0.);

    // ********************************************************************************

    const Real* plo = Geom(lev).ProbLo();
    const Real* phi = Geom(lev).ProbHi();

    const Box& domain = Geom(lev).Domain();

    int do_move = 1;

 {
    BL_PROFILE("ParticleContainer::moveKickDriftPart1()");

    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {

        AoS& particles = pti.GetArrayOfStructs();
        int Np = particles.size();

        //std::cout << "---Np=" << Np << std::endl;

        if (Np > 0) 
        {
           const Box& state_box = (*state_ptr)[pti].box();
           const Box&   src_box = (*tmp_src_ptr)[pti].box();

           update_particles(&Np,&lev, particles.data(), 
                            (*state_ptr)[pti].dataPtr(),
                            state_box.loVect(), state_box.hiVect(),
                            (*tmp_src_ptr)[pti].dataPtr(),
                            src_box.loVect(), src_box.hiVect(),
                            domain.loVect(), domain.hiVect(),
                            plo,phi,reflect_lo,reflect_hi,dx,dt,&do_move);
        }
    }
  }
    //load balancing
    //the weight w should be the estimate time per particle times paritcle count
    /* {
      if (PeleC::do_particle_load_balance) {
        //for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
        //  get_new_data(Work_Estimate_Type)[mfi].plus(w);        
        //}
      }
    }*/

    // ********************************************************************************
    // Make sure the momentum put into ghost cells of each grid is added to both 
    //      valid regions AND the ghost cells of other grids.  If at level = 0 
    //      we can accomplish this with SumBoundary; however if this level has ghost
    //      cells not covered at this level then we need to do more.
    // ********************************************************************************
  {
    BL_PROFILE("ParticleContainer::moveKickDriftPart2()");
    {
      BL_PROFILE("ParticleContainer::moveKickDriftPart2.1()");

    if (lev > 0)
    {
       int ncomp = tmp_src_ptr->nComp();
       MultiFab tmp(this->m_gdb->ParticleBoxArray(lev), 
                    this->m_gdb->ParticleDistributionMap(lev),
                    ncomp, tmp_src_ptr->nGrow());
       tmp.setVal(0.);
       tmp.copy(*tmp_src_ptr, 0, 0, ncomp, tmp_src_ptr->nGrow(), tmp_src_ptr->nGrow(), 
                Geom(lev).periodicity(), FabArrayBase::ADD);
       tmp_src_ptr->copy(tmp, 0, 0, ncomp, tmp_src_width, tmp_src_width, 
                         Geom(lev).periodicity(), FabArrayBase::COPY);
    } else {
       tmp_src_ptr->SumBoundary(Geom(lev).periodicity());
    }
    }
    {
      BL_PROFILE("ParticleContainer::moveKickDriftPart2.2()");
    // Add new sources into source *after* we have called SumBoundary
    MultiFab::Add(source,*tmp_src_ptr,0,0,source.nComp(),std::min(source.nGrow(),tmp_src_ptr->nGrow()));
    delete tmp_src_ptr;

    // Fill ghost cells after we've synced up ..
    source.FillBoundary(Geom(lev).periodicity()); 

    // Only delete this if in fact we created it.  Note we didn't change state_ptr so 
    //  we don't need to copy anything back into state
    if (state_ptr != &state ) delete state_ptr;
    }
  }

    // ********************************************************************************
  {
    BL_PROFILE("ParticleContainer::moveKickDriftPart3()");

    if (lev > 0 && sub_cycle)
    {
        ParticleLocData pld;
        for (auto& kv : pmap) {
            AoS&  pbox       = kv.second.GetArrayOfStructs();
            const int   n    = pbox.size();

#ifdef _OPENMP
#pragma omp parallel for private(pld)
#endif
            for (int i = 0; i < n; i++)
            {
                ParticleType& p = pbox[i];
                if (p.id() <= 0) continue;

                // Move the particle to the proper ghost cell.
                //      and remove any *ghost* particles that have gone too far
                // Note that this should only negate ghost particles, not real particles.
                if (!this->Where(p, pld, lev, lev, where_width))
                {
                    // Assert that the particle being removed is a ghost particle;
                    // the ghost particle is no longer in relevant ghost cells for this grid.
                    if (p.id() == GhostParticleID)
                    {
                        p.id() = -1;
                    }
                    else
                    {
                        std::cout << "Oops -- removing particle " << p.id() << " " << p.pos(0) << " " << p.pos(1) << std::endl;
                        p.id() = -1;
                        //amrex::Error("Trying to get rid of a non-ghost particle in moveKickDrift");
                    }
                }
            }
        }
    }
  }

    // ********************************************************************************

    if (this->m_verbose > 1)
    {
        Real stoptime = ParallelDescriptor::second() - strttime;

        ParallelDescriptor::ReduceRealMax(stoptime,ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor())
        {
            std::cout << "SprayParticle,Container::moveKickDrift() time: " << stoptime << '\n';
        }
    }
}

void
SprayParticleContainer::moveKick(MultiFab& state,
                                MultiFab& source,
                                int              lev,
                                Real      dt,
                                int              tmp_src_width)
{
    const Real* dx = Geom(lev).CellSize();

    BL_PROFILE("ParticleContainer::moveKick()");
    BL_ASSERT(lev >= 0);
    //std::cout << " GROW -2" << state.nGrow() << std::endl;
    BL_ASSERT(state.nGrow() >= 2);

    //If there are no particles at this level
    if (lev >= this->GetParticles().size())
        return;

    const Real strttime      = ParallelDescriptor::second();

    MultiFab* state_ptr;
    MultiFab* tmp_src_ptr;

    // ********************************************************************************
    // We only make a new state_ptr if the boxArray of the state differs from the
    // boxArray onto which the particles are decomposed
    // ********************************************************************************

    if (this->OnSameGrids(lev, state))
    {
        state_ptr = &state;
    }
    else
    {
        state_ptr = new MultiFab(this->m_gdb->ParticleBoxArray(lev),
                                       this->m_gdb->ParticleDistributionMap(lev),
                                       state.nComp(),state.nGrow());
        state_ptr->setVal(0.);
        state_ptr->copy(state,0,0,state.nComp());
        state_ptr->FillBoundary(Geom(lev).periodicity()); 
    }

    // ********************************************************************************
    // We make a temporary MultiFab for the source here and initialize it to zero
    // because if we use one that already has values in it, the SumBoundary call
    // will end up duplicating those values with each call
    // ********************************************************************************

    tmp_src_width = 3; // HACK?

    tmp_src_ptr = new MultiFab(this->m_gdb->ParticleBoxArray(lev),
                                      this->m_gdb->ParticleDistributionMap(lev),
                                      source.nComp(),tmp_src_width);
    tmp_src_ptr->setVal(0.);

    // ********************************************************************************

    const Real* plo = Geom(lev).ProbLo();
    const Real* phi = Geom(lev).ProbHi();

    const Box& domain = Geom(lev).Domain();

    int do_move = 0;

    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {

        AoS& particles = pti.GetArrayOfStructs();
        int Np = particles.size();

        if (Np > 0) 
        {
           const Box& state_box = (*state_ptr)[pti].box();
           const Box&   src_box = (*tmp_src_ptr)[pti].box();

           update_particles(&Np,&lev, particles.data(),
                            (*state_ptr)[pti].dataPtr(),
                            state_box.loVect(), state_box.hiVect(),
                            (*tmp_src_ptr)[pti].dataPtr(),
                            src_box.loVect(), src_box.hiVect(),
                            domain.loVect(), domain.hiVect(),
                            plo,phi,reflect_lo,reflect_hi,
                            dx,dt,&do_move);
        }
    }

    // ********************************************************************************
    // Make sure the momentum put into ghost cells of each grid is added to both
    //      valid regions AND the ghost cells of other grids.  If at level = 0
    //      we can accomplish this with SumBoundary; however if this level has ghost
    //      cells not covered at this level then we need to do more.
    // ********************************************************************************
    if (lev > 0)
    {
       int ncomp = tmp_src_ptr->nComp();
       int ngrow = source.nGrow();
       MultiFab tmp(this->m_gdb->ParticleBoxArray(lev),
                    this->m_gdb->ParticleDistributionMap(lev),
                    ncomp, ngrow);
       tmp.setVal(0.);
       tmp.copy(*tmp_src_ptr, 0, 0, ncomp, ngrow, ngrow, Geom(lev).periodicity(), FabArrayBase::ADD);
       tmp_src_ptr->copy(tmp, 0, 0, ncomp, ngrow, ngrow, Geom(lev).periodicity(), FabArrayBase::COPY);
    } else {
       tmp_src_ptr->SumBoundary(Geom(lev).periodicity());
    }

    // Add new sources into source *after* we have called SumBoundary
    MultiFab::Add(source,*tmp_src_ptr,0,0,source.nComp(),source.nGrow());
    delete tmp_src_ptr;

    // Fill ghost cells after we've synced up ..
    source.FillBoundary(Geom(lev).periodicity());

    // Only delete this if in fact we created it.  Note we didn't change state_ptr so
    //  we don't need to copy anything back into state
    if (state_ptr != &state ) delete state_ptr;

    // ********************************************************************************

    if (this->m_verbose > 1)
    {
        Real stoptime = ParallelDescriptor::second() - strttime;

        ParallelDescriptor::ReduceRealMax(stoptime,ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor())
        {
            std::cout << "SprayParticleContainer::moveKick() time: " << stoptime << '\n';
        }
    }
}

Real
SprayParticleContainer::estTimestep (int lev, Real cfl) const
{
    Real dt = 1.e50;

    if (this->GetParticles().size() == 0)
        return dt;

    const Real      strttime         = ParallelDescriptor::second();
    const Geometry& geom             = this->m_gdb->Geom(lev);
    const Real*     dx               = geom.CellSize();
    int   tnum                       = 1;

#ifdef _OPENMP
    tnum = omp_get_max_threads();
#endif

    Vector<Real> ldt(tnum,1e50);

    long num_particles_at_level = 0;
    Real max_mag_vel_over_dx;

    // We assume here that the velocity components follow mass in the rdata array
    for (MyParConstIter pti(*this, lev); pti.isValid(); ++pti) 
    {
      auto& particles = pti.GetArrayOfStructs();
      size_t Np = pti.numParticles();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (unsigned i = 0; i < Np; ++i)
      {
          const ParticleType& p = particles[i];
          if (p.id() > 0) {
            const Real mag_vel_over_dx[BL_SPACEDIM] = { D_DECL(std::abs(p.rdata(1))/dx[0],
                                                               std::abs(p.rdata(2))/dx[1],
                                                               std::abs(p.rdata(3))/dx[2]) };
            max_mag_vel_over_dx = std::max(mag_vel_over_dx[0], mag_vel_over_dx[1]);
#if (BL_SPACEDIM > 2)
            max_mag_vel_over_dx = std::max(mag_vel_over_dx[2], max_mag_vel_over_dx);
#endif
          }

       Real dt_part = (max_mag_vel_over_dx > 0) ? (cfl / max_mag_vel_over_dx) : 1e50;

       int tid = 0;
#ifdef _OPENMP
       tid = omp_get_thread_num();
#endif
      ldt[tid] = std::min(dt_part, ldt[tid]);
      }
      num_particles_at_level += Np;
    }

    for (int i = 0; i < ldt.size(); i++)
        dt = std::min(dt, ldt[i]);

    ParallelDescriptor::ReduceRealMin(dt);

    //
    // Set dt negative if there are no particles at this level.
    //
    ParallelDescriptor::ReduceLongSum(num_particles_at_level);

    if (num_particles_at_level == 0) dt = -1.e50;

    if (this->m_verbose > 1)
    {
        Real stoptime = ParallelDescriptor::second() - strttime;
        ParallelDescriptor::ReduceRealMax(stoptime,ParallelDescriptor::IOProcessorNumber());
        if (ParallelDescriptor::IOProcessor())
            std::cout << "SprayParticleContainer::estTimestep() time: " << stoptime << '\n';
    }

    return dt;
}

void
SprayParticleContainer::insertParticles (Real time, int nstep, int lev)
{
  const Geometry& geom = this->m_gdb->Geom(lev);

  for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {

    auto& particles = GetParticles(lev)[std::make_pair(mfi.index(),
                                                       mfi.LocalTileIndex())];

    const Box& bx = mfi.tilebox();
    const RealBox& temp = RealBox(bx,geom.CellSize(),geom.ProbLo());
    const Real* xlo = temp.lo();
    const Real* xhi = temp.hi();

// Take random draw from inside the jet of x width 0.02 and length
// geometry.prob_lo     =   -0.12 -0.06 0.0
// geometry.prob_hi     =  0.12 0.06 0.24

    for (int iter = 0; iter < 10; iter++)
    {
    Real x = 0.04*(rand()%100)/99.-0.02;
    Real y = 0.12*(rand()%100)/99.-0.06;
    Real z = 0.24*(rand()%100)/99.;

// Check if injection point belongs to box
   if(x > xlo[0] && x < xhi[0] && y > xlo[1] && y < xhi[1] && z > xlo[2] && z < xhi[2]) {

     std::cout << " new particle at " <<x<< " " <<y<< " " <<z<< '\n' << std::endl;

     ParticleType p;
     p.id()   = ParticleType::NextID();
     p.cpu()  = ParallelDescriptor::MyProc();

     p.pos(0) = x;
     p.pos(1) = y;
     p.pos(2) = z;

     // fill in NSR_SPR items
     p.rdata(0) = 0.;
     p.rdata(1) = 0.;
     if (AMREX_SPACEDIM>2) {
       p.rdata(2) = 0.;
     }
     p.rdata(AMREX_SPACEDIM) = 293.; // temperature
     p.rdata(AMREX_SPACEDIM+1) = 0.0010; // diameter
     p.rdata(AMREX_SPACEDIM+2) = 0.68141; // fuel density
    
     particles.push_back(p);
   }
  } // for iter
 }
  Redistribute();
}

static bool my_first = true; // Hack to avoid adding new particle on every call
static Real t_next = 0.;

void
SprayParticleContainer::insertParticles (Real time, int nstep, int lev, amrex::Real dt)
{
  if (my_first) {
    srand(150);
    my_first = false;
    t_next = time;
  }

  if(nstep%10!=0) return; 
  //return;
  //std::cout<<"---insertParticles called---\n"<<std::endl;

  //std::cout << "TIME " << time << " at parent step "<< nstep <<"; next injection time at " << t_next << '\n' << std::endl;
  //if(time < t_next) return;
  t_next+= 1e-5;

  const double pi = std::atan(1.0)*4.;
  const Geometry& geom = this->m_gdb->Geom(lev);

  int i = 0;
  
  Real part_D    = 1e-6;
  Real part_rho  = 681.41;
  Real part_mass = part_rho * 4./3. * pi * pow(part_D*.5,3.);
  Real mass_flow = 2e-15/10.;
  Real mass_target = mass_flow * dt;
  Real mass = 0.;

  int n_max_part = 100000;

  int n = 0;
 
  // number of particles to insert overall
  int n_part_insert = 33;
  //int n_part_insert = 105;
  
  //amrex::Print() << "mass_flow="<<mass_flow<<", mass_target="<<mass_target<<", part_mass="<<part_mass<<", mass_target/part_mass="<<(mass_target/part_mass)<<", n_part_insert="<<n_part_insert<<'\n';
  //std::cout << "parts[0].x=" << parts[0].x << ", parts[0].y=" << parts[0].y << "parts[1].x=" << parts[1].x << ", parts[1].y=" << parts[1].y << '\n';

  for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {

    auto& particles = GetParticles(lev)[std::make_pair(mfi.index(),
                                                       mfi.LocalTileIndex())];

    const Box& bx = mfi.tilebox();
    const RealBox& temp = RealBox(bx,geom.CellSize(),geom.ProbLo());
    const Real* xlo = temp.lo();
    const Real* xhi = temp.hi();

    for(int n = 0, iter=0; n < n_part_insert && iter<2000 ;) {
      Real rand1 = double(rand()) / (double(RAND_MAX) + 1.0);
      Real rand2 = double(rand()) / (double(RAND_MAX) + 1.0);

	    ParticleType p;

      p.pos(0) = 1e-6;
      p.pos(1) = 2.0+4.0*rand1;
      //p.pos(1) = 3.25+1.0*rand1;
      p.pos(2) = 1.5*rand2;

      if(p.pos(0) > xlo[0] && p.pos(0) < xhi[0] && 
         p.pos(1) > xlo[1] && p.pos(1) < xhi[1] && 
         p.pos(2) > xlo[2] && p.pos(2) < xhi[2]) {

	      p.id()   = ParticleType::NextID();
	      p.cpu()  = ParallelDescriptor::MyProc();
	  
	      // fill in NSR_SPR items; p.rdata(0) to p.rdata(2) are the velocities
	      p.rdata(0) = 1.0;
	      p.rdata(1) = 0.0;
	      p.rdata(2) = 0.0;

	      p.rdata(AMREX_SPACEDIM) = 293.; // temperature
	      p.rdata(AMREX_SPACEDIM+1) = 1e-6; // diameter
	      p.rdata(AMREX_SPACEDIM+2) = 681.41; // fuel density

	      particles.push_back(p);

        n++;
      }
      iter++;
	//std::cout << "Inserting particle p.id=" << p.id()<<'\n' << std::endl;
	//std::cout << "Inserting particle p.id=" << p.id() << ", p.cpu="<< p.cpu() <<", time=" << time << ", level=" << lev << ", nstep=" << nstep <<", xlo[0]="<< xlo[0] << x << ", xhi[0]="<<xhi[0]<<", xlo[1]="<<xlo[1]<<", xhi[1]="<<xhi[1]<<", xlo[2]="<<xlo[2]<<", xhi[2]="<<xhi[2]<<'\n' << std::endl;
   }
   // std::cout << " wge particle count size=" << particles.size()<<'\n' << std::endl;
 }

  Redistribute();
}

void
SprayParticleContainer::injectParticles (Real time, int nstep, int lev)
{
// Specialized for injection trough a 2D orifice - many values are hardcoded
  if (my_first) {
    srand(15);
    my_first = false;
    t_next = time;
  }

  std::cout << "TIME " << time << " at parent step "<< nstep <<"; next injection time at " << t_next << '\n' << std::endl;
  if(time < t_next) return;
  t_next+= 1e-5;

  const Real r0 = 1.5;
  const Real y = -30;
  const Real y0 = r0*3.7320; // virtual origin for 15 deg angle

  const Geometry& geom = this->m_gdb->Geom(lev);

  for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {

    auto& particles = GetParticles(lev)[std::make_pair(mfi.index(),
                                                       mfi.LocalTileIndex())];

    const Box& bx = mfi.tilebox();
    const RealBox& temp = RealBox(bx,geom.CellSize(),geom.ProbLo());
    const Real* xlo = temp.lo();
    const Real* xhi = temp.hi();

// Check if injection rake belongs to box
    if(xlo[0] <= r0 && xhi[0] >= -r0 && xlo[1] <= -30 && xhi[1] >= -30) {

      Real x = (rand()%10)/3.-r0;
      if (x > xlo[0] && x < xhi[0]) {

        ParticleType p;
        p.id()   = ParticleType::NextID();
        p.cpu()  = ParallelDescriptor::MyProc();

        p.pos(0) = x;
        p.pos(1) = y;
        if (AMREX_SPACEDIM>2) {
          p.pos(2) = 0.;
        }

        Real angle = atan(x/y0); // seen from virtual origin (0,-y0)
        // fill in NSR_SPR items
        p.rdata(0) = 1000.*sin(angle); // ux
        p.rdata(1) = 1000.*cos(angle); // uy
        if (AMREX_SPACEDIM>2) {
          p.rdata(2) = 0.0;
        }
        p.rdata(AMREX_SPACEDIM) = 293.; // temperature
        p.rdata(AMREX_SPACEDIM+1) = 0.0200; // diameter
        p.rdata(AMREX_SPACEDIM+2) = 0.68141; // fuel density
    
        particles.push_back(p);
      }
    }
  }

  Redistribute();
} 
