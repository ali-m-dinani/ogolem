package org.ogolem.adaptive;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.ogolem.core.GlobalConfig;

class AdaptiveConfOperatorsTest {

  @Test
  void parsesOperatorProbabilitiesAndParentSelection() throws Exception {
    final AdaptiveConf configuration =
        new AdaptiveConf(
            new String[] {
              "AdaptivableChoice=externalcaller:3D",
              "CrossProbability=0.8",
              "MutationRatio=0.2",
              "ParentsChoice=fitnessrankbased:gausswidth=0.3"
            },
            new GlobalConfig());
    assertEquals(0.8, configuration.crossPossibility);
    assertEquals(0.2, configuration.mutationRatio);
    assertEquals("fitnessrankbased:gausswidth=0.3", configuration.whichParentsChoice);
  }

  @Test
  void rejectsInvalidProbabilities() {
    for (final String keyword : new String[] {"CrossProbability", "MutationRatio"}) {
      for (final String value : new String[] {"-0.1", "1.1", "NaN", "Infinity"}) {
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new AdaptiveConf(
                    new String[] {"AdaptivableChoice=externalcaller:3D", keyword + "=" + value},
                    new GlobalConfig()));
      }
    }
  }
}
