#pragma once

#include "Surface.h"

namespace Pivot {

	class ImplicitSphere : public Surface {
	public:
		ImplicitSphere(Vector3d const &center, double radius) : m_Center { center }, m_Radius { radius } { }

		virtual Vector3d ClosestPositionOf(Vector3d const &pos) const override { return m_Center + ClosestNormalOf(pos) * m_Radius; }
		virtual Vector3d ClosestNormalOf  (Vector3d const &pos) const override { return (pos - m_Center).normalized(); }
		virtual double   SignedDistanceTo (Vector3d const &pos) const override { return (pos - m_Center).norm() - m_Radius; }

	private:
		Vector3d m_Center;
		double   m_Radius;
	};

	class ImplicitDisk : public Surface {
	public:
		ImplicitDisk(Vector3d const &center, Vector3d const &direct, double radius, double roundRadius) : m_Center { center }, m_Radius { radius }, m_Direct { direct.normalized() }, m_RoundRadius { roundRadius } { }

		// TODO
		virtual Vector3d ClosestPositionOf(Vector3d const &pos) const override { return Vector3d::Zero(); }
		// TODO
		virtual Vector3d ClosestNormalOf  (Vector3d const &pos) const override { return Vector3d::Zero(); }
		virtual double   SignedDistanceTo (Vector3d const &pos) const override {
			Vector3d dpos = (pos - m_Center);
			Vector3d dposProj = (dpos - dpos.dot(m_Direct) * m_Direct);
			Vector3d cpDisk = m_Center + dposProj.normalized() * (std::min)(m_Radius, dposProj.norm());
			return (pos - cpDisk).norm() - m_RoundRadius;
		}

	private:
		Vector3d m_Center;
		Vector3d m_Direct;
		double   m_Radius;
		double   m_RoundRadius;
	};

	class ImplicitScrew : public Surface {
	public:
		ImplicitScrew (Vector3d const &center, double height1, double height2, double r1, double r2, double angle, double k) : m_Center { center }, m_Height1 { height1 }, m_Height2 { height2 }, m_R1 { r1 }, m_R2 { r2 }, m_Angle { angle }, m_K { k } { }
		// TODO
		virtual Vector3d ClosestNormalOf  (Vector3d const &pos) const override { return Vector3d::Zero(); }
		virtual double SignedDistanceTo(Vector3d const &pos) const override {
			Vector3d dpos = pos - m_Center;
			double cone_factor = (std::min)(1.0, abs(dpos.y()) / m_Height2);
			double c = cos(m_K * dpos.y());
			double s = sin(m_K * dpos.y());
			Matrix2d mat = (Matrix2d() << c, -s, s, c).finished();
			Vector2d pos2d = mat * Vector2d(dpos.x(), dpos.z());
			Vector3d tdpos(pos2d.x(), dpos.y(), pos2d.y());

			Vector2d C(sin(m_Angle), cos(m_Angle));
			pos2d.x() = abs(pos2d.x());
			double l = pos2d.norm() - m_R1 * cone_factor;
			double m = (pos2d - C * (std::min)((std::max)(pos2d.dot(C), 0.0), m_R1 * cone_factor)).norm();
			m *= (C.y() * pos2d.x() - C.x()* pos2d.y() < 0) ? -1 : 1;
			double d1 = (std::max)(l, m);

			double d2 = pos2d.norm() - m_R1 * cone_factor;
			double d3 = pos2d.norm() - m_R2 * cone_factor;
			double d = (std::min)((std::max)(d2, -d1), d3);

			Vector2d w(d, abs(dpos.y()) - m_Height2);
			double discrew = (std::min)((std::max)(w.x(), w.y()), 0.0) + (w.cwiseMax(0)).norm();
			return (std::max)(dpos.y() + m_Height1, discrew);
		}

	private:
		Vector3d m_Center;
		double m_Height1;
		double m_Height2;
		double m_Angle;
		double m_K;
		double m_R1;
		double m_R2;
	};
	
	class ImplicitBox : public Surface {
	public:
		ImplicitBox(Vector3d const &minCorner, Vector3d const &lengths) : m_Center { minCorner + lengths / 2 }, m_HalfLengths { lengths / 2 } { }

		virtual Vector3d ClosestNormalOf(Vector3d const &pos) const override {
			Vector3d const phi = (pos - m_Center).cwiseAbs() - m_HalfLengths;
			Vector3d normal;
			if ((phi.array() <= 0).all()) {
				int axis;
				phi.maxCoeff(&axis);
				normal = Vector3d::Unit(axis);
			} else {
				normal = phi.cwiseMax(0);
			}
			return normal.cwiseProduct((pos - m_Center).cwiseSign()).normalized();
		}

		virtual double SignedDistanceTo(Vector3d const &pos) const override {
			Vector3d const phi = (pos - m_Center).cwiseAbs() - m_HalfLengths;
			if ((phi.array() <= 0).all()) {
				return phi.maxCoeff();
			} else {
				return phi.cwiseMax(0).norm();
			}
		}

	private:
		Vector3d m_Center;
		Vector3d m_HalfLengths;
	};

	class ImplicitPlane : public Surface {
	public:
		ImplicitPlane(Vector3d const &position, Vector3d const &direction) : m_Position(position), m_Normal(direction.normalized()) { }

		virtual Vector3d ClosestNormalOf (Vector3d const &pos) const override { return m_Normal; }
		virtual double   SignedDistanceTo(Vector3d const &pos) const override { return (pos - m_Position).dot(m_Normal); }

	private:
		Vector3d const m_Position;
		Vector3d const m_Normal;
	};

	class ImplicitEllipsoid : public Surface {
	public:

		ImplicitEllipsoid(Vector3d const &center, Vector3d const &semiAxels) : m_Center(center), m_SemiAxels(semiAxels) { }

		virtual Vector3d ClosestPositionOf(Vector3d const &pos) const override { auto pos_ = pos - m_Center; return pos_ / pos_.cwiseQuotient(m_SemiAxels).norm(); } // not accurate solution
		virtual Vector3d ClosestNormalOf  (Vector3d const &pos) const override { auto pos_ = pos - m_Center; return (pos_ - ClosestPositionOf(pos_)).normalized() * (Surrounds(pos) ? -1 : 1); }
		virtual double   DistanceTo       (Vector3d const &pos) const override { auto pos_ = pos - m_Center; return (pos_ - ClosestPositionOf(pos_)).norm(); }
		virtual double   SignedDistanceTo (Vector3d const &pos) const override { auto pos_ = pos - m_Center; return DistanceTo(pos_) * (Surrounds(pos_) ? -1 : 1); }
		virtual bool     Surrounds        (Vector3d const &pos) const override { auto pos_ = pos - m_Center; return pos_.cwiseQuotient(m_SemiAxels).squaredNorm() <= 1; }

	private:
		Vector3d m_Center;
		Vector3d m_SemiAxels;
	};
}
