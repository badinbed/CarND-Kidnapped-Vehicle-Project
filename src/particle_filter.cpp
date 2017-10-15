/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // set number of particles
    num_particles = 500;
    particles.clear();
    particles.reserve(num_particles);


    // normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // initalize particles with random gaussian position and pose around the GPS location and weight to 1
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;

        particles.push_back(p);
    }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    // update all particle states according to the bicylce motion model
    for(auto p = particles.begin(); p != particles.end(); ++p) {
        double x = p->x;
        double y = p->y;
        double theta = p->theta;

        // to avoid division by zero we check for the yaw rate and handle according to the bike motion model
        if(abs(yaw_rate) < 0.0001) {
            x += velocity * delta_t * cos(theta);
            y += velocity * delta_t * sin(theta);
        } else {
            x += velocity/yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
            y += velocity/yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
            theta += yaw_rate * delta_t;
        }

        // normal distributions for x, y and theta
        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);

        // add noise and update particle state
        p->x = dist_x(gen);
        p->y = dist_y(gen);
        p->theta = dist_theta(gen);
    }
}

double ParticleFilter::findLandmarkAssociation(LandmarkObs* predicted, const Map &map_landmarks) {

    // -1 in case there is no landmark
    predicted->id = -1;
    double minDistance = std::numeric_limits<double>::max();
    for(auto lm = map_landmarks.landmark_list.begin(); lm != map_landmarks.landmark_list.end(); ++lm) {
        // euclidean distance to landmark
        double dx = predicted->x - lm->x_f;
        double dy = predicted->y - lm->y_f;
        double d = sqrt(dx*dx + dy*dy);

        // associate obs with landmark if it is closer than the current minimum
        if(d < minDistance) {
            predicted->id = lm->id_i;
            minDistance = d;
        }
    }
    return minDistance;
}

void ParticleFilter::updateWeights(double sensor_range, double std_lm[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    double gaussianNorm = 1.0/(2*M_PI*std_lm[0]*std_lm[1]);
    weights.clear();
    for(auto p = particles.begin(); p != particles.end(); ++p) {

        // reset particle
        p->weight = 1.0;
        p->associations.clear();
        p->sense_x.clear();
        p->sense_y.clear();

        // calculate weight over all observations
        for(auto o = observations.begin(); o != observations.end(); ++o) {
            // transform observations into world coordinates
            LandmarkObs obsWorld;
            obsWorld.x = p->x + cos(p->theta)*o->x - sin(p->theta)*o->y;
            obsWorld.y = p->y + sin(p->theta)*o->x + cos(p->theta)*o->y;

            // find nearest landmark
            double lmDistance = findLandmarkAssociation(&obsWorld, map_landmarks);


            // landmarks further away than sensor_range will be ignored
            double dx = sqrt(sensor_range);
            double dy = sqrt(sensor_range);
            if(lmDistance <= sensor_range) {
                auto& lmd = map_landmarks.landmark_list[obsWorld.id - 1];
                dx = obsWorld.x - lmd.x_f;
                dy = obsWorld.y - lmd.y_f;
            }

            // calculate weight for this observation
            double w = gaussianNorm * exp(-((dx*dx)/(2*std_lm[0]*std_lm[0]) + (dy*dy)/(2*std_lm[1]*std_lm[1])));
            p->weight *= w;

            // update associations for whatever
            p->associations.push_back(obsWorld.id);
            p->sense_x.push_back(obsWorld.x);
            p->sense_y.push_back(obsWorld.y);
        }
        weights.push_back(p->weight);
    }
}

void ParticleFilter::resample() {

    vector<Particle> resampled_particles;
    resampled_particles.reserve(num_particles);

    // init discrete distribution with particle weights and draw num_particles new particles
    std::discrete_distribution<int> distr(weights.begin(), weights.end());
    for(int i = 0; i < num_particles; ++i) {
        resampled_particles.push_back(particles[distr(gen)]);
    }

    particles.assign(resampled_particles.begin(), resampled_particles.end());

}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
