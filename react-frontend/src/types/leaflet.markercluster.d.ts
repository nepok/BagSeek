import 'leaflet';

declare module 'leaflet' {
  namespace L {
    interface MarkerClusterGroupOptions {
      maxClusterRadius?: number;
      spiderfyOnMaxZoom?: boolean;
      showCoverageOnHover?: boolean;
      zoomToBoundsOnClick?: boolean;
      iconCreateFunction?: (cluster: MarkerCluster) => Icon | DivIcon;
    }

    interface MarkerCluster extends Layer {
      getChildCount(): number;
    }

    interface MarkerClusterGroup extends LayerGroup {
      addLayer(layer: Layer): this;
      removeLayer(layer: Layer): this;
    }
  }

  function markerClusterGroup(options?: L.MarkerClusterGroupOptions): L.MarkerClusterGroup;
}

declare module 'leaflet.markercluster';

