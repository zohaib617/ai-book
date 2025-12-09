import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/modules/module-1-ros2/intro">
            Start Reading - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="An Educational Book on Robotics, AI, and Humanoid Systems">
      <HomepageHeader />
      <main>
        <section className={styles.modules}>
          <div className="container padding-vert--lg">
            <div className="row">
              <div className="col col--3">
                <div className="card">
                  <div className="card__header text--center">
                    <h3>Module 1</h3>
                  </div>
                  <div className="card__body text--center">
                    <p><strong>ROS 2: The Robotic Nervous System</strong></p>
                    <p>Learn about nodes, topics, services, and Python integration</p>
                  </div>
                  <div className="card__footer text--center">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/modules/module-1-ros2/intro">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card">
                  <div className="card__header text--center">
                    <h3>Module 2</h3>
                  </div>
                  <div className="card__body text--center">
                    <p><strong>Digital Twin (Gazebo & Unity)</strong></p>
                    <p>Simulation, physics, and sensor modeling</p>
                  </div>
                  <div className="card__footer text--center">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/modules/module-2-digital-twin/intro">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card">
                  <div className="card__header text--center">
                    <h3>Module 3</h3>
                  </div>
                  <div className="card__body text--center">
                    <p><strong>AI-Robot Brain (NVIDIA Isaac™)</strong></p>
                    <p>Perception, navigation, and SLAM systems</p>
                  </div>
                  <div className="card__footer text--center">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/modules/module-3-ai-brain/intro">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--3">
                <div className="card">
                  <div className="card__header text--center">
                    <h3>Module 4</h3>
                  </div>
                  <div className="card__body text--center">
                    <p><strong>Vision-Language-Action (VLA)</strong></p>
                    <p>Voice-to-action and cognitive planning</p>
                  </div>
                  <div className="card__footer text--center">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/modules/module-4-vla/intro">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}