using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Jet.CLR;

namespace clr_tests
{
    [TestClass]
    public class AnimationTests
    {
        [TestMethod]
        public void TestConstructors()
        {
            NoOpAnimation anim = new NoOpAnimation();
            Frame frame = new Frame();
            Assert.AreEqual(frame.Index, anim.LastFrame.Index);
            Assert.AreEqual(frame.TimeIntervalInSeconds, anim.LastFrame.TimeIntervalInSeconds);
        }

        [TestMethod]
        public void TestUpdate()
        {
            NoOpAnimation anim = new NoOpAnimation();
            Frame frame = new Frame(10, 0.01);

            anim.Update(frame);

            Assert.AreEqual(frame.Index, anim.LastFrame.Index);
            Assert.AreEqual(frame.TimeIntervalInSeconds, anim.LastFrame.TimeIntervalInSeconds);
        }

        [TestMethod]
        public void TestUpdateWithRunner()
        {
            NoOpAnimation anim = new NoOpAnimation();
            AnimationRunner runner = new AnimationRunner();

            runner.Run(anim, 10.0, 25);

            Assert.AreEqual(24, anim.LastFrame.Index);
            Assert.AreEqual(0.1, anim.LastFrame.TimeIntervalInSeconds);
        }
    }
}
